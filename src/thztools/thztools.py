from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.fft import irfft, rfft, rfftfreq
from numpy.typing import ArrayLike
import scipy.linalg as la
from scipy import odr
from scipy.optimize import minimize


def fftfreq(n, ts):
    """Computes the positive and negative frequencies sampled in the FFT.

    Parameters
    ----------
    n : int
        Number of time samples
    ts: float
        Sampling time

    Returns
    -------
    f : ndarray
        Frequency vector (1/``ts``) of length n containing the sample
        frequencies.
    """

    if n % 2 == 1:
        f = np.fft.fftfreq(n, ts)
    else:
        f = np.fft.fftfreq(n, ts)
        f[int(n / 2)] = -f[int(n / 2)]

    return f


def noisevar(sigma: ArrayLike, mu: ArrayLike, ts: float) -> ArrayLike:
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


def noiseamp(sigma: ArrayLike, mu: ArrayLike, ts: float) -> ArrayLike:
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
) -> tuple[ArrayLike, ArrayLike]:
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
) -> ArrayLike:
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
        np.fft.rfft(x) * np.exp(1j * phase), n=n
    ) / np.expand_dims(a, axis=a.ndim)

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
) -> ArrayLike:
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
    n = xx.shape[0]
    w = 2 * np.pi * rfftfreq(n, ts)
    h_f = np.conj(fun(theta, w))

    ry = yy - irfft(rfft(xx) * h_f, n=n)
    vy = np.diag(sigmay**2)

    h_imp = irfft(h_f, n=n)
    h = la.circulant(h_imp)

    # For V_xx = diag(sigma_x ** 2),
    # uy = h @ ((sigmax ** 2) * h).T
    #    = h @ diag(sigmax ** 2) @ h.T
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
) -> ArrayLike:
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
    n = xx.shape[-1]
    delta = (xx - mu) / sigmax
    w = 2 * np.pi * rfftfreq(n, ts)
    h_f = fun(theta, w)

    eps = (yy - irfft(rfft(mu) * h_f, n=n)) / sigmay

    res = np.concatenate((delta, eps))

    return res


def tdtf(fun: Callable, theta: ArrayLike, n: int, ts: float) -> ArrayLike:
    """
    Computes the time-domain transfer matrix for a frequency response function.

    Parameters
    ----------
        fun : callable
            Frequency function, in the form fun(theta, w), where theta
            is a vector of the function parameters. The function should be
            expressed in the +iwt convention and must be Hermitian.

        theta : array_like
            Input parameters for the function.

        n : int
            Number of time samples.

        ts : array_like
            Sampling time.

    Returns
    -------
        h : array_like
            Transfer matrix with size (n,n).

    """

    f = rfftfreq(n, ts)
    w = 2 * np.pi * f
    tfunp = fun(theta, w)

    imp = irfft(tfunp, n=n)
    h = la.circulant(imp).T

    return h


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
) -> tuple[ArrayLike, ArrayLike]:
    r"""
    Compute negative log-likelihood for the time-domain noise model.

    Computes the negative log-likelihood function for obtaining the
    data matrix ``x``, given the parameter dictionary param.

    Parameters
    ----------
    x : array_like
        Data matrix.
    mu : array_like
        Signal vector with length N.
    logv : array_like
        Array of three noise parameters.
    a: array_like
        Amplitude vector with length M.
    eta : array_like
        Delay vector with length M.
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

    def fun(_, _w):
        return -1j * _w

    tdtf(fun, 0, n, ts)

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

    # Compute gradient if requested
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
    ignore_a: bool = True,
    ignore_eta: bool = True,
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
        Initial guess, noise model parameters with size (3,).
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
    ignore_a : bool, optional
        Amplitude vector.
    ignore_eta : bool, optional
        Delay vector.

    Returns
    --------
    p : dict
        Output parameter dictionary containing:
            eta : ndarray
                Delay vector.
            a : ndarray
                Amplitude vector.
            mu : ndarray
                Signal vector.
            var : ndarray
                Log of noise parameters
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
    # Parse and validate function inputs
    x = np.asarray(x)
    if x.ndim != 2:
        msg = "Data array x must be 2D."
        raise ValueError(msg)
    n, m = x.shape

    if v0 is None:
        v0 = np.mean(np.var(x, 1)) * np.array([1, 1, 1])
    else:
        v0 = np.asarray(v0)
        if v0.size != 3:
            msg = "Noise parameter array logv must have 3 elements."
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
        eta0 = np.ones(m)
    else:
        eta0 = np.asarray(eta0)
        if eta0.size != m:
            msg = "Size of eta0 is incompatible with data array x."
            raise ValueError(msg)

    mle = {"x0": np.array([])}
    idxstart = 0
    idxrange = {}

    # If fix['logv'], return log(v0); otherwise return logv parameters
    if fix_v:

        def setplogv(_):
            return np.log(v0)

    else:
        mle["x0"] = np.concatenate((mle["x0"], np.log(v0)))
        idxend = idxstart + 3
        idxrange["logv"] = np.arange(idxstart, idxend)

        def setplogv(_p):
            return _p[idxrange["logv"]]

        idxstart = idxend

    # If Fix['mu'], return mu0, otherwise, return mu parameters
    if fix_mu:

        def setpmu(_):
            return mu0

    else:
        mle["x0"] = np.concatenate((mle["x0"], mu0))
        idxend = idxstart + n
        idxrange["mu"] = np.arange(idxstart, idxend)

        def setpmu(_p):
            return _p[idxrange["mu"]]

        idxstart = idxend
    pass

    # If ignore_a, return None; if fix_a, return a0; if (!fix_a & fix_mu),
    # return all a parameters; if !fix_a & !fix_mu, return all a parameters
    # but first

    if ignore_a:

        def setpa(_):
            return None

    elif fix_a:

        def setpa(_):
            return a0

    elif fix_mu:
        mle["x0"] = np.concatenate((mle["x0"], a0))
        idxend = idxstart + m
        idxrange["a"] = np.arange(idxstart, idxend)

        def setpa(_p):
            return _p[idxrange["a"]]

        idxstart = idxend
    else:
        mle["x0"] = np.concatenate((mle["x0"], a0[1:] / a0[0]))
        idxend = idxstart + m - 1
        idxrange["a"] = np.arange(idxstart, idxend)

        def setpa(_p):
            return np.concatenate(([1], _p[idxrange["a"]]), axis=0)

        idxstart = idxend
    pass

    # If ignore_eta, return None; if fix_eta, return eta0; if !fix_eta &
    # fix_mu,return all eta parameters; if !fix_eta & !fix_mu, return all eta
    # parameters but first

    if ignore_eta:

        def setpeta(_):
            return None

    elif fix_eta:

        def setpeta(_):
            return eta0

    elif fix_mu:
        mle["x0"] = np.concatenate((mle["x0"], eta0))
        idxend = idxstart + m
        idxrange["eta"] = np.arange(idxstart, idxend)

        def setpeta(_p):
            return _p[idxrange["eta"]]

    else:
        mle["x0"] = np.concatenate((mle["x0"], eta0[1:] - eta0[0]))
        idxend = idxstart + m - 1
        idxrange["eta"] = np.arange(idxstart, idxend)

        def setpeta(_p):
            return np.concatenate(([0], _p[idxrange["eta"]]), axis=0)

    pass

    def fun(_, _w):
        return -1j * _w

    tdtf(fun, 0, n, ts)

    def parsein(_p):
        return {
            "logv": setplogv(_p),
            "mu": setpmu(_p),
            "a": setpa(_p),
            "eta": setpeta(_p),
            "ts": ts,
        }

    def objective(_p):
        p_dict = parsein(_p)
        _mu = p_dict.pop("mu")
        _logv = p_dict.pop("logv")
        return tdnll(
            x.T,
            _mu,
            _logv,
            **p_dict,
            fix_logv=fix_v,
            fix_mu=fix_mu,
            fix_a=fix_a,
            fix_eta=fix_eta,
        )

    mle["objective"] = objective
    out = minimize(mle["objective"], mle["x0"], method="BFGS", jac=True)

    # The trust-region algorithm returns the Hessian for the next-to-last
    # iterate, which may not be near the final point. To check, test for
    # positive definiteness by attempting to Cholesky factorize it. If it
    # returns an error, rerun the optimization with the quasi-Newton algorithm
    # from the current optimal point.

    try:
        la.cholesky(la.inv(out.hess_inv))
        hess = la.inv(out.hess_inv)
    except la.LinAlgError:
        print(
            "Hessian returned by FMINUNC is not positive definite;\n"
            "recalculating with quasi-Newton algorithm"
        )

        mle["x0"] = out.x
        out2 = minimize(mle["objective"], mle["x0"], method="BFGS", jac=True)
        hess = la.inv(out2.hess_inv)

    # Parse output
    p = {}
    idxrange = {}
    idxstart = 0

    if fix_v:
        p["var"] = v0
    else:
        idxend = idxstart + 3
        idxrange["logv"] = np.arange(idxstart, idxend)
        idxstart = idxend
        p["var"] = np.exp(out.x[idxrange["logv"]])
    pass

    if fix_mu:
        p["mu"] = mu0
    else:
        idxend = idxstart + n
        idxrange["mu"] = np.arange(idxstart, idxend)
        idxstart = idxend
        p["mu"] = out.x[idxrange["mu"]]
    pass

    if ignore_a or fix_a:
        p["a"] = a0
    elif fix_mu:
        idxend = idxstart + m
        idxrange["a"] = np.arange(idxstart, idxend)
        idxstart = idxend
        p["a"] = out.x[idxrange["a"]]
    else:
        idxend = idxstart + m - 1
        idxrange["a"] = np.arange(idxstart, idxend)
        idxstart = idxend
        p["a"] = np.concatenate(([1], out.x[idxrange["a"]]), axis=0)
    pass

    if ignore_eta or fix_eta:
        p["eta"] = eta0
    elif fix_mu:
        idxend = idxstart + m
        idxrange["eta"] = np.arange(idxstart, idxend)
        p["eta"] = out.x[idxrange["eta"]]
    else:
        idxend = idxstart + m - 1
        idxrange["eta"] = np.arange(idxstart, idxend)
        p["eta"] = np.concatenate(([0], out.x[idxrange["eta"]]), axis=0)
    pass

    p["ts"] = ts

    vary_param = np.logical_not(
        [
            fix_v,
            fix_mu,
            fix_a or ignore_a,
            fix_eta or ignore_eta,
        ]
    )
    diagnostic = {
        "grad": out.jac,
        "hessian": hess,
        "err": {"var": [], "mu": [], "a": [], "eta": []},
    }
    v = np.dot(np.eye(hess.shape[0]), la.inv(hess))
    err = np.sqrt(np.diag(v))
    idxstart = 0
    if vary_param[0]:
        diagnostic["err"]["var"] = np.sqrt(
            np.diag(np.diag(p["var"]) * v[0:3, 0:3]) * np.diag(p["var"])
        )
        idxstart = idxstart + 3
    pass

    if vary_param[1]:
        diagnostic["err"]["mu"] = err[idxstart : idxstart + n]
        idxstart = idxstart + n
    pass

    if vary_param[2]:
        if vary_param[1]:
            diagnostic["err"]["a"] = err[idxstart : idxstart + m - 1]
            idxstart = idxstart + m - 1
        else:
            diagnostic["err"]["a"] = err[idxstart : idxstart + m]
            idxstart = idxstart + m
    pass

    if vary_param[3]:
        if vary_param[1]:
            diagnostic["err"]["eta"] = err[idxstart : idxstart + m - 1]
        else:
            diagnostic["err"]["eta"] = err[idxstart : idxstart + m]
    pass

    return [p, out.fun, diagnostic]
