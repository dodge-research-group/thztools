from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from numpy.fft import irfft, rfft, rfftfreq
from numpy.typing import ArrayLike
from scipy import signal
from scipy.optimize import approx_fprime as fprime
from scipy.optimize import minimize

NUM_NOISE_PARAMETERS = 3
NUM_NOISE_DATA_DIMENSIONS = 2


def noisevar(sigma: ArrayLike, mu: ArrayLike, ts: float) -> np.ndarray:
    r"""
    Compute the time-domain noise variance.

    Parameters
    ----------
    sigma : array_like
        Noise parameter array with shape (3, ). The first element corresponds
        to the amplitude noise, in signal units (ie, the same units as ``mu``);
        the second element corresponds to multiplicative noise, which is
        dimensionless; and the third element corresponds to timebase noise, in
        units of signal/time, where the units for time are the same as for
        ``ts``.
    mu :  array_like
        Time-domain signal.
    ts : float
        Sampling time.

    Returns
    -------
    ndarray
        Time-domain noise variance.
    """
    sigma = np.asarray(sigma)
    mu = np.asarray(mu)

    n = mu.shape[0]
    w = 2 * np.pi * rfftfreq(n, ts)
    mudot = irfft(1j * w * rfft(mu), n=n)

    return sigma[0] ** 2 + (sigma[1] * mu) ** 2 + (sigma[2] * mudot) ** 2


def noiseamp(sigma: ArrayLike, mu: ArrayLike, ts: float) -> np.ndarray:
    r"""
    Compute the time-domain noise amplitude.

    Parameters
    ----------
    sigma : array_like
        Noise parameter array with shape (3, ). The first element corresponds
        to the amplitude noise, in signal units (ie, the same units as mu);
        the second element corresponds to multiplicative noise, which is
        dimensionless; and the third element corresponds to timebase noise, in
        units of signal/time, where the units for time are the same as for t.
    mu :  array_like
        Time-domain signal.
    ts : float
        Sampling time.

    Returns
    -------
    ndarray
        Time-domain noise amplitude, in signal units.

    """

    return np.sqrt(noisevar(sigma, mu, ts))


def thzgen(
    n: int,
    ts: float,
    t0: float,
    *,
    a: float = 1.0,
    taur: float = 0.3,
    tauc: float = 0.1,
    fwhm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Simulate a terahertz pulse.

    Parameters
    ----------

    n : int
        Number of samples.

    ts : float
        Sampling time.

    t0 : float
        Pulse center.

    a : float, optional
        Peak amplitude.

    taur : float, optional
        Current pulse rise time.

    tauc : float, optional
        Current pulse decay time.

    fwhm : float, optional
        Laser pulse FWHM.

    Returns
    -------

    ndarray
        Signal array.

    ndarray
        Array of time samples.

    """
    taul = fwhm / np.sqrt(2 * np.log(2))

    f = rfftfreq(n, ts)

    w = 2 * np.pi * f
    ell = np.exp(-((w * taul) ** 2) / 2) / np.sqrt(2 * np.pi * taul**2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (ell * r) ** 2 * np.exp(1j * w * t0)

    t2 = ts * np.arange(n)

    y = irfft(np.conj(s), n=n)
    y = a * y / np.max(y)

    return y, t2


def scaleshift(
    x: ArrayLike,
    *,
    a: ArrayLike | None = None,
    eta: ArrayLike | None = None,
    ts: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    """Rescale and shift signal vectors.

    Parameters
    ----------
    x : array_like
        Data array.
    a : array_like, optional
        Scale array.
    eta : array_like, optional
        Shift array.
    ts : float, optional
        Sampling time. Default is 1.0.
    axis : int, optional
        Axis over which to apply the correction. If not given, applies over the
        last axis in ``x``.

    Returns
    -------
    xadj : ndarray
        Adjusted data array.

    """
    x = np.asarray(x)
    if x.size == 0:
        return np.empty(x.shape)

    axis = int(axis)
    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)

    n = x.shape[-1]
    m = x.shape[:-1]

    if a is None:
        a = np.ones(m)
    else:
        a = np.asarray(a)
        if a.shape != m:
            msg = (
                f"Scale correction with shape {a.shape} can not be applied "
                f"to data with shape {x.shape}"
            )
            raise ValueError(msg)

    if eta is None:
        eta = np.zeros(m)
    else:
        eta = np.asarray(eta)
        if eta.shape != m:
            msg = (
                f"Shift correction with shape {a.shape} can not be applied "
                f"to data with shape {x.shape}"
            )
            raise ValueError(msg)

    f = rfftfreq(n, ts)
    w = 2 * np.pi * f
    phase = np.expand_dims(eta, axis=eta.ndim) * w

    xadj = np.fft.irfft(
        np.fft.rfft(x) * np.exp(-1j * phase), n=n
    ) * np.expand_dims(a, axis=a.ndim)

    if x.ndim > 1:
        if axis != -1:
            xadj = np.moveaxis(xadj, -1, axis)

    return xadj


def costfunlsq(
    fun: Callable,
    theta: ArrayLike,
    xx: ArrayLike,
    yy: ArrayLike,
    sigmax: ArrayLike,
    sigmay: ArrayLike,
    ts: float,
) -> np.ndarray:
    r"""Computes the residual vector for the maximum likelihood cost function.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        theta : array_like
            Input parameters for the function.

        xx : array_like
            Measured input signal.

        yy : array_like
            Measured output signal.

        sigmax : array_like
            Noise vector of the input signal.

        sigmay : array_like
            Noise vector of the output signal.

        ts : float
            Sampling time.

    Returns
    -------
    res : array_like


    """
    theta = np.asarray(theta)
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    sigmax = np.asarray(sigmax)
    sigmay = np.asarray(sigmay)

    n = xx.shape[0]
    w = 2 * np.pi * rfftfreq(n, ts)
    h_f = np.conj(fun(theta, w))

    ry = yy - irfft(rfft(xx) * h_f, n=n)
    vy = np.diag(sigmay**2)

    h_imp = irfft(h_f, n=n)
    h = la.circulant(h_imp)

    # For V_xx = diag(sigma_x ** 2),
    # uy = h @ ((sigmax ** 2) * h).T
    # is equivalent to and faster than
    # uy = h @ diag(sigmax ** 2) @ h.T
    uy = h @ ((sigmax**2) * h).T

    res = la.inv(la.sqrtm(uy + vy)) @ ry

    return res


def costfuntls(
    fun: Callable,
    theta: ArrayLike,
    mu: ArrayLike,
    xx: ArrayLike,
    yy: ArrayLike,
    sigmax: ArrayLike,
    sigmay: ArrayLike,
    ts: float,
) -> np.ndarray:
    r"""Computes the residual vector for the total least squares cost function.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), +iwt convention.

        theta : array_like
            Input parameters for the function.

        mu : array_like
            Estimated input signal.

        xx : array_like
            Measured input signal.

        yy : array_like
            Measured output signal.

        sigmax : array_like
            Noise vector of the input signal.

        sigmay : array_like
            Noise vector of the output signal.

        ts : float
            Sampling time.

    Returns
    -------
    res : array_like


    """
    theta = np.asarray(theta)
    mu = np.asarray(mu)
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    sigmax = np.asarray(sigmax)
    sigmay = np.asarray(sigmay)

    n = xx.shape[-1]
    delta_norm = (xx - mu) / sigmax
    w = 2 * np.pi * rfftfreq(n, ts)
    h_f = fun(theta, w)

    eps_norm = (yy - irfft(rfft(mu) * h_f, n=n)) / sigmay

    res = np.concatenate((delta_norm, eps_norm))

    return res


def tdnll(
    x: ArrayLike,
    mu: ArrayLike,
    logv: ArrayLike,
    a: ArrayLike,
    eta: ArrayLike,
    ts: float,
    *,
    fix_logv: bool,
    fix_mu: bool,
    fix_a: bool,
    fix_eta: bool,
) -> tuple[float, np.ndarray]:
    r"""
    Compute negative log-likelihood for the time-domain noise model.

    Computes the negative log-likelihood function for obtaining the
    data matrix `x` given `mu`, `logv`, `a`, and `eta`.

    Parameters
    ----------
    x : array_like
        Data matrix.
    mu : array_like
        Signal vector with shape (n,).
    logv : array_like
        Array of three noise parameters.
    a: array_like
        Amplitude vector with shape (m,).
    eta : array_like
        Delay vector with shape (m,).
    ts : float
        Sampling time.
    fix_logv : bool
        Exclude noise parameters from gradiate calculation when ``True``.
    fix_mu : bool
        Exclude signal vector from gradiate calculation when ``True``.
    fix_a : bool
        Exclude amplitude vector from gradiate calculation when ``True``.
    fix_eta : bool
        Exclude delay vector from gradiate calculation when ``True``.

    Returns
    -------
    nll : float
        Negative log-likelihood.
    gradnll : array_like
        Gradient of the negative log-likelihood function with respect to free
        parameters.
    """
    x = np.asarray(x)
    logv = np.asarray(logv)
    mu = np.asarray(mu)
    a = np.asarray(a)
    eta = np.asarray(eta)

    m, n = x.shape

    # Compute variance
    v = np.exp(logv)

    # Compute frequency vector and Fourier coefficients of mu
    f = rfftfreq(n, ts)
    w = 2 * np.pi * f
    mu_f = rfft(mu)

    exp_iweta = np.exp(1j * np.outer(eta, w))
    zeta_f = ((np.conj(exp_iweta) * mu_f).T * a).T
    zeta = irfft(zeta_f, n=n)

    # Compute negative - log likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Alternative case: A, eta, or both are not set to defaults
    dzeta = irfft(1j * w * zeta_f, n=n)

    valpha = v[0]
    vbeta = v[1] * zeta**2
    vtau = v[2] * dzeta**2
    vtot = valpha + vbeta + vtau

    resnormsq = ressq / vtot
    nll = (
        m * n * np.log(2 * np.pi) / 2
        + np.sum(np.log(vtot)) / 2
        + np.sum(resnormsq) / 2
    )

    # Compute gradient
    gradnll = np.array([])
    if not (fix_logv & fix_mu & fix_a & fix_eta):
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot**2
        if not fix_logv:
            # Gradient wrt logv
            gradnll = np.append(gradnll, 0.5 * np.sum(dvar) * v[0])
            gradnll = np.append(gradnll, 0.5 * np.sum(zeta**2 * dvar) * v[1])
            gradnll = np.append(
                gradnll, 0.5 * np.sum(dzeta**2 * dvar) * v[2]
            )
        if not fix_mu:
            # Gradient wrt mu
            p = rfft(v[1] * dvar * zeta - reswt) - 1j * v[2] * w * rfft(
                dvar * dzeta
            )
            gradnll = np.append(
                gradnll, np.sum((irfft(exp_iweta * p, n=n).T * a).T, axis=0)
            )
        if not fix_a:
            # Gradient wrt A
            term = (vtot - valpha) * dvar - reswt * zeta
            dnllda = np.sum(term, axis=1).T / a
            # Exclude first term for consistency with MATLAB version
            gradnll = np.append(gradnll, dnllda[1:])
        if not fix_eta:
            # Gradient wrt eta
            ddzeta = irfft(-(w**2) * zeta_f, n=n)
            dnlldeta = -np.sum(
                dvar * (zeta * dzeta * v[1] + dzeta * ddzeta * v[2])
                - reswt * dzeta,
                axis=1,
            )
            # Exclude first term for consistency with MATLAB version
            gradnll = np.append(gradnll, dnlldeta[1:])

    return nll, gradnll


def tdnoisefit(
    x: ArrayLike,
    *,
    v0: ArrayLike | None = None,
    mu0: ArrayLike | None = None,
    a0: ArrayLike | None = None,
    eta0: ArrayLike | None = None,
    ts: float = 1.0,
    fix_v: bool = False,
    fix_mu: bool = False,
    fix_a: bool = True,
    fix_eta: bool = True,
) -> tuple[dict, float, dict]:
    r"""
    Compute time-domain noise model parameters.

    Computes the noise parameters sigma and the underlying signal vector ``mu``
    for the data matrix ``x``, where the columns of ``x`` are each noisy
    measurements of ``mu``.

    Parameters
    ----------
    x : ndarray
        Data array.
    v0 : ndarray, optional
        Initial guess, noise model parameters with size (3,), expressed as
        variance amplitudes.
    mu0 : ndarray, optional
        Initial guess, signal vector with size (n,).
    a0 : ndarray, optional
        Initial guess, amplitude vector with size (m,).
    eta0 : ndarray, optional
        Initial guess, delay vector with size (m,).
    ts : float, optional
        Sampling time
    fix_v : bool, optional
        Noise variance parameters.
    fix_mu : bool, optional
        Signal vector.
    fix_a : bool, optional
        Amplitude vector.
    fix_eta : bool, optional
        Delay vector.

    Returns
    --------
    p : dict
        Output parameter dictionary containing:
            var : ndarray
                Noise parameters, expressed as variance amplitudes.
            mu : ndarray
                Signal vector.
            a : ndarray
                Amplitude vector.
            eta : ndarray
                Delay vector.
    fval : float
        Value of NLL cost function from FMINUNC
    Diagnostic : dict
        Dictionary containing diagnostic information
            err : dic
                Dictionary containing  error of the parameters.
            grad : ndarray
                Negative loglikelihood cost function gradient from
                scipy.optimize.minimize BFGS method.
            hessian : ndarray
                Negative loglikelihood cost function hessian from
                scipy.optimize.minimize BFGS method.
    """
    if fix_v and fix_mu and fix_a and fix_eta:
        msg = "All variables are fixed"
        raise ValueError(msg)
    # Parse and validate function inputs
    x = np.asarray(x)
    if x.ndim != NUM_NOISE_DATA_DIMENSIONS:
        msg = "Data array x must be 2D"
        raise ValueError(msg)
    n, m = x.shape

    if v0 is None:
        v0 = np.mean(np.var(x, 1)) * np.array([1, 1, 1])
    else:
        v0 = np.asarray(v0)
        if v0.size != NUM_NOISE_PARAMETERS:
            msg = (
                "Noise parameter array logv must have "
                f"{NUM_NOISE_PARAMETERS} elements."
            )
            raise ValueError(msg)

    if mu0 is None:
        mu0 = np.mean(x, 1)
    else:
        mu0 = np.asarray(mu0)
        if mu0.size != n:
            msg = "Size of mu0 is incompatible with data array x."
            raise ValueError(msg)

    if a0 is None:
        a0 = np.ones(m)
    else:
        a0 = np.asarray(a0)
        if a0.size != m:
            msg = "Size of a0 is incompatible with data array x."
            raise ValueError(msg)

    if eta0 is None:
        eta0 = np.zeros(m)
    else:
        eta0 = np.asarray(eta0)
        if eta0.size != m:
            msg = "Size of eta0 is incompatible with data array x."
            raise ValueError(msg)

    # Set initial guesses for all free parameters
    x0 = np.array([])
    if not fix_v:
        x0 = np.concatenate((x0, np.log(v0)))
    if not fix_mu:
        x0 = np.concatenate((x0, mu0))
    if not fix_a:
        x0 = np.concatenate((x0, a0[1:] / a0[0]))
    if not fix_eta:
        x0 = np.concatenate((x0, eta0[1:] - eta0[0]))

    # Bundle free parameters together into objective function
    def objective(_p):
        if fix_v:
            _logv = np.log(v0)
        else:
            _logv = _p[:3]
            _p = _p[3:]
        if fix_mu:
            _mu = mu0
        else:
            _mu = _p[:n]
            _p = _p[n:]
        if fix_a:
            _a = a0
        else:
            _a = np.concatenate((np.array([1.0]), _p[: m - 1]))
            _p = _p[m - 1 :]
        if fix_eta:
            _eta = eta0
        else:
            _eta = np.concatenate((np.array([0.0]), _p[: m - 1]))
        return tdnll(
            x.T,
            _mu,
            _logv,
            _a,
            _eta,
            ts,
            fix_logv=fix_v,
            fix_mu=fix_mu,
            fix_a=fix_a,
            fix_eta=fix_eta,
        )

    # Minimize cost function with respect to free parameters
    out = minimize(objective, x0, method="BFGS", jac=True)

    # Parse output
    p = {}
    x_out = out.x
    if fix_v:
        p["var"] = v0
    else:
        p["var"] = np.exp(x_out[:3])
        x_out = x_out[3:]

    if fix_mu:
        p["mu"] = mu0
    else:
        p["mu"] = x_out[:n]
        x_out = x_out[n:]

    if fix_a:
        p["a"] = a0
    else:
        p["a"] = np.concatenate(([1], x_out[: m - 1]))
        x_out = x_out[m - 1 :]

    if fix_eta:
        p["eta"] = eta0
    else:
        p["eta"] = np.concatenate(([0], x_out[: m - 1]))

    p["ts"] = ts

    diagnostic = {
        "grad": out.jac,
        "hess_inv": out.hess_inv,
        "err": {
            "var": np.array([]),
            "mu": np.array([]),
            "a": np.array([]),
            "eta": np.array([]),
        },
    }
    err = np.sqrt(np.diag(out.hess_inv))
    if not fix_v:
        # Propagate error from log(V) to V
        diagnostic["err"]["var"] = np.sqrt(
            np.diag(np.diag(p["var"]) @ out.hess_inv[0:3, 0:3])
            @ np.diag(p["var"])
        )
        err = err[3:]

    if not fix_mu:
        diagnostic["err"]["mu"] = err[:n]
        err = err[n:]

    if not fix_a:
        diagnostic["err"]["a"] = np.concatenate(([0], err[: m - 1]))
        err = err[m - 1 :]

    if not fix_eta:
        diagnostic["err"]["eta"] = np.concatenate(([0], err[: m - 1]))

    return p, out.fun, diagnostic


def fit(
    fun: Callable,
    p0: ArrayLike,
    xx: ArrayLike,
    yy: ArrayLike,
    *,
    ts: float = 1,
    noise_parms: ArrayLike = (1, 0, 0),
    p_bounds: ArrayLike | None = None,
    jac: Callable | None = None,
    args: ArrayLike = (),
    kwargs: dict | None = None,
) -> dict:
    r"""
    Fit THz time-domain data to a transfer function.

    Computes the noise on the input `xx` and output `yy` time series using
    `noiseamp`. Then uses the total residuals generated by `costfuntls` to fit
    the input and output to the transfer function.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w,*args,**kwargs), +iwt
            convention.
        p0 : array_like
            Initial guess for the theta.
        xx : array_like
            Measured input signal.
        yy : array_like
            Measured output signal.
        ts : float, optional
            Sampling time.
        noise_parms : None or array_like, optional
            Noise parameters with size (3,), expressed as standard deviation
            amplitudes.
        p_bounds : None, 2-tuple of array_like, or Bounds, optional
            Lower and upper bounds on fit parameter(s).
        jac : None or callable, optional
            Method of calculating derivative of the output signal residuals
            with respect to the fit parameter(s), theta.
        args : tuple, optional
            Additional arguments passed to `fun` and `jac`.
        kwargs : dict, optional
            Additional keyword arguments passed to `fun` and `jac`.

    Returns
    -------
        p : dict
            Output parameter dictionary containing:
                p_opt : array_like
                    Optimal fit parameters.
                p_cov : array_like
                    Variance of p_opt.
                mu_opt : array_like
                    Optimal underlying waveform.
                mu_var : array_like
                    Variance of mu_opt.
                resnorm : float
                    The value of $\chi^2$.
                delta : array_like
                    Residuals of the input waveform `xx`.
                epsilon : array_like
                    Resiuals of the output waveform `yy`.
                success : bool
                    True if one of the convergence criteria is satisfied.
    """
    fit_method = "trf"

    p0 = np.asarray(p0)
    xx = np.asarray(xx)
    yy = np.asarray(yy)

    n = yy.shape[-1]
    n_p = len(p0)

    if p_bounds is None:
        p_bounds = (-np.inf, np.inf)
        fit_method = "lm"
    elif len(p_bounds) == 2:  # noqa: PLR2004
        p_bounds = (
            np.concatenate((p_bounds[0], np.full((n,), -np.inf))),
            np.concatenate((p_bounds[1], np.full((n,), np.inf))),
        )
    else:
        msg = "`bounds` must contain 2 elements."
        raise ValueError(msg)

    if kwargs is None:
        kwargs = {}

    w = 2 * np.pi * rfftfreq(n, ts)
    n_f = len(w)
    sigma_x = noiseamp(noise_parms, xx, ts=ts)
    sigma_y = noiseamp(noise_parms, yy, ts=ts)
    p0_est = np.concatenate((p0, np.ones(n)))

    def function(_theta, _w):
        return fun(_theta, _w, *args, **kwargs)

    def function_flat(_x):
        _tf = function(_x, w)
        return np.concatenate((np.real(_tf), np.imag(_tf)))

    def td_fun(_p, _x):
        _h = irfft(rfft(_x) * function(_p, w), n=n)
        return _h

    def jacobian(_p):
        if jac is None:
            _tf_prime = fprime(_p, function_flat)
            return _tf_prime[0:n_f] + 1j * _tf_prime[n_f:]
        else:
            return jac(_p, w, *args, **kwargs)

    def jac_fun(_x):
        p_est = _x[:n_p]
        mu_est = xx[:] - _x[n_p:]
        jac_tl = np.zeros((n, n_p))
        jac_tr = np.diag(1 / sigma_x)
        jac_bl = -(
            irfft(rfft(mu_est) * np.atleast_2d(jacobian(p_est)).T, n=n)
            / sigma_y
        ).T
        jac_br = (
            la.circulant(td_fun(p_est, signal.unit_impulse(n))).T / sigma_y
        ).T
        jac_tot = np.block([[jac_tl, jac_tr], [jac_bl, jac_br]])
        return jac_tot

    result = opt.least_squares(
        lambda _p: costfuntls(
            function,
            _p[:n_p],
            xx[:] - _p[n_p:],
            xx[:],
            yy[:],
            sigma_x,
            sigma_y,
            ts,
        ),
        p0_est,
        jac=jac_fun,
        bounds=p_bounds,
        method=fit_method,
        x_scale=np.concatenate((np.ones(n_p), sigma_x)),
    )

    # Parse output
    p = {}
    p["p_opt"] = result.x[:n_p]
    p["mu_opt"] = xx - result.x[n_p:]
    _, s, vt = la.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    p["p_var"] = np.diag(np.dot(vt.T / s**2, vt))[:n_p]
    p["mu_var"] = np.diag(np.dot(vt.T / s**2, vt))[n_p:]
    p["resnorm"] = 2 * result.cost
    p["delta"] = xx - p["mu_opt"]
    p["epsilon"] = yy - irfft(rfft(p["mu_opt"]) * function(p["p_opt"], w), n=n)
    p["success"] = result.success
    return p
