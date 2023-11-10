from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from numpy.fft import irfft, rfft, rfftfreq
from numpy.random import default_rng
from numpy.typing import ArrayLike
from scipy import signal
from scipy.optimize import approx_fprime as fprime
from scipy.optimize import minimize

NUM_NOISE_PARAMETERS = 3
NUM_NOISE_DATA_DIMENSIONS = 2


@dataclass
class NoiseResult:
    r"""
    Represents the noise parameter estimate output.

    Parameters
    ----------
    p : dict
        Output parameter dictionary containing the following.

            var : ndarray, shape (3,)
                Noise parameters, expressed as variance amplitudes.
            mu : ndarray, shape (n,)
                Signal vector.
            a : ndarray, shape (m,)
                Amplitude vector.
            eta : ndarray, shape (m,)
                Delay vector.
    fval : float
        Value of NLL cost function from FMINUNC.
    diagnostic : dict
        Dictionary containing diagnostic information:

            grad_scaled : ndarray
                Gradient of the scaled negative log-likelihood function with
                respect to the scaled fit parameters.
            hess_inv_scaled : ndarray
                Inverse of the Hessian obtained from scipy.optimize.minimize
                using the BFGS method, which is determined for the scaled
                negative log-likelihood function with respect to the scaled
                fit parameters.
            err : dict
                Dictionary containing  error of the parameters. Uses the same
                keys as ``p``.
            success : bool
                Whether the fit terminated successfully.
            status : int
                Termination status of fit.
            message : str
                Description of the termination condition.
            nfev, njev : int
                Number of evaluations of the objective function and the
                Jacobian.
            nit : int
                Number of iterations performed by `scipy.optimize.minimize`.
    """
    p: dict
    fval: float
    diagnostic: dict


# noinspection PyShadowingNames
@dataclass
class NoiseModel:
    r"""
    Terahertz noise model class.

    For noise parameters :math:`\sigma_\alpha`, :math:`\sigma_\beta`,
    :math:`\sigma_\tau` and signal vector :math:`\boldsymbol{\mu}`, the
    :math:`k`-th element of the time-domain noise variance
    :math:`\boldsymbol{\sigma}^2` is given by [1]_

    .. math:: \sigma_k^2 = \sigma_\alpha^2 + \sigma_\beta^2\mu_k^2 \
        + \sigma_\tau^2(\mathbf{D}\boldsymbol{\mu})_k^2,

    where :math:`\mathbf{D}` is the time-domain derivative operator.

    Parameters
    ----------
    alpha : float
        Additive noise amplitude.
    beta : float
        Multiplicative noise amplitude.
    tau : float
        Timebase noise amplitude.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default set to None, for which
        the ``tau`` parameter is given in units of the sampling time.

    References
    ----------
    .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
        Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
        likelihood parameter estimation in terahertz time-domain
        spectroscopy," Opt. Express **29**, 4912-4926 (2021),
        `<https://doi.org/10.1364/OE.417724>`_.

    Examples
    --------
    The following example shows the noise variance :math:`\sigma^2(t)` for
    noise parameters :math:`\sigma_\alpha = 10^{-4}`,
    :math:`\sigma_\beta = 10^{-2}`, :math:`\sigma_\tau = 10^{-3}` and the
    simulated signal :math:`\mu(t)`. The signal amplitude is normalized to
    its peak magnitude, :math:`\mu_0`. The noise variance is normalized to
    :math:`(\sigma_\beta\mu_0)^2`.

    .. plot::
       :include-source: True

        >>> import matplotlib.pyplot as plt
        >>> import thztools as thz
        >>> n, dt, t0 = 256, 0.05, 2.5
        >>> mu, t = thz.wave(n, dt, t0)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_mod = thz.NoiseModel(alpha=alpha, beta=beta, tau=tau,
        ... dt=dt)
        >>> var_t = noise_mod.variance(mu)

        >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        >>> axs[0].plot(t, var_t / beta**2)
        >>> axs[0].set_ylabel(r"$\sigma^2/(\sigma_\beta\mu_0)^2$")
        >>> axs[1].plot(t, mu)
        >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
        >>> axs[1].set_xlabel("t (ps)")
        >>> plt.show()
    """
    alpha: float
    beta: float
    tau: float
    dt: float | None = None

    # noinspection PyShadowingNames
    def variance(self, x: ArrayLike, *, axis: int = -1) -> np.ndarray:
        r"""
        Compute the time-domain noise variance.

        Parameters
        ----------
        x :  array_like
            Time-domain signal.
        axis : int, optional
            Axis over which to apply the correction. If not given, applies over
            the last axis in ``x``.

        Returns
        -------
        var_t : ndarray
            Time-domain noise variance, in signal units (squared).

        Notes
        -----
        For noise parameters :math:`\sigma_\alpha`, :math:`\sigma_\beta`,
        :math:`\sigma_\tau` and signal vector :math:`\boldsymbol{\mu}`, the
        :math:`k`-th element of the time-domain noise variance
        :math:`\boldsymbol{\sigma}^2` is given by [1]_

        .. math:: \sigma_k^2 = \sigma_\alpha^2 + \sigma_\beta^2\mu_k^2 \
            + \sigma_\tau^2(\mathbf{D}\boldsymbol{\mu})_k^2,

        where :math:`\mathbf{D}` is the time-domain derivative operator.

        References
        ----------
        .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
            Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
            likelihood parameter estimation in terahertz time-domain
            spectroscopy," Opt. Express **29**, 4912-4926 (2021),
            `<https://doi.org/10.1364/OE.417724>`_.

        Examples
        --------
        The following example shows the noise variance :math:`\sigma^2(t)` for
        noise parameters :math:`\sigma_\alpha = 10^{-4}`,
        :math:`\sigma_\beta = 10^{-2}`, :math:`\sigma_\tau = 10^{-3}` and the
        simulated signal :math:`\mu(t)`. The signal amplitude is normalized to
        its peak magnitude, :math:`\mu_0`. The noise variance is normalized to
        :math:`(\sigma_\beta\mu_0)^2`.

        .. plot::
           :include-source: True

            >>> import matplotlib.pyplot as plt
            >>> import thztools as thz
            >>> n, dt, t0 = 256, 0.05, 2.5
            >>> mu, t = thz.wave(n, dt, t0)
            >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
            >>> noise_mod = thz.NoiseModel(alpha=alpha, beta=beta, tau=tau,
            ... dt=dt)
            >>> var_t = noise_mod.variance(mu)

            >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
            >>> axs[0].plot(t, var_t / beta**2)
            >>> axs[0].set_ylabel(r"$\sigma^2/(\sigma_\beta\mu_0)^2$")
            >>> axs[1].plot(t, mu)
            >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
            >>> axs[1].set_xlabel("t (ps)")
            >>> plt.show()
        """
        dt = self.dt
        if dt is None:
            dt = 1.0
        x = np.asarray(x)
        axis = int(axis)
        if x.ndim > 1:
            if axis != -1:
                x = np.moveaxis(x, axis, -1)

        n = x.shape[-1]
        w_scaled = 2 * np.pi * rfftfreq(n)
        xdot = irfft(1j * w_scaled * rfft(x), n=n) / dt

        var_t = self.alpha**2 + (self.beta * x) ** 2 + (self.tau * xdot) ** 2

        if x.ndim > 1:
            if axis != -1:
                var_t = np.moveaxis(var_t, -1, axis)

        return var_t

    # noinspection PyShadowingNames
    def amplitude(self, x: ArrayLike, *, axis: int = -1) -> np.ndarray:
        r"""
        Compute the time-domain noise amplitude.

        Parameters
        ----------
        x :  array_like
            Time-domain signal.
        axis : int, optional
            Axis over which to apply the correction. If not given, applies over
            the last axis in ``x``.

        Returns
        -------
        sigma_t : ndarray
            Time-domain noise amplitude, in signal units.

        Notes
        -----
        For noise parameters :math:`\sigma_\alpha`, :math:`\sigma_\beta`,
        :math:`\sigma_\tau` and signal vector :math:`\boldsymbol{\mu}`, the
        :math:`k`-th element of the time-domain noise amplitude vector
        :math:`\boldsymbol{\sigma}` is given by [1]_

        .. math:: \sigma_k = \sqrt{\sigma_\alpha^2 + \sigma_\beta^2\mu_k^2 \
            + \sigma_\tau^2(\mathbf{D}\boldsymbol{\mu})_k^2},

        where :math:`\mathbf{D}` is the time-domain derivative operator.

        References
        ----------
        .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
            Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
            likelihood parameter estimation in terahertz time-domain
            spectroscopy," Opt. Express **29**, 4912-4926 (2021),
            `<https://doi.org/10.1364/OE.417724>`_.

        Examples
        --------
        The following example shows the noise amplitude :math:`\sigma(t)` for
        noise parameters :math:`\sigma_\alpha = 10^{-4}`,
        :math:`\sigma_\beta = 10^{-2}`, :math:`\sigma_\tau = 10^{-3}` and the
        simulated signal :math:`\mu(t)`. The signal amplitude is normalized to
        its peak magnitude, :math:`\mu_0`. The noise amplitude is normalized to
        :math:`\sigma_\beta\mu_0`.

        .. plot::
           :include-source: True

            >>> import matplotlib.pyplot as plt
            >>> import thztools as thz

            >>> n, dt, t0 = 256, 0.05, 2.5
            >>> mu, t = thz.wave(n, dt, t0)
            >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
            >>> noise_mod = thz.NoiseModel(alpha=alpha, beta=beta, tau=tau,
            ... dt=dt)
            >>> sigma_t = noise_mod.amplitude(mu)

            >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
            >>> axs[0].plot(t, sigma_t / beta)
            >>> axs[0].set_ylabel(r"$\sigma/(\sigma_\beta\mu_0)$")
            >>> axs[1].plot(t, mu)
            >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
            >>> axs[1].set_xlabel("t (ps)")
            >>> plt.show()
        """
        return np.sqrt(self.variance(x, axis=axis))

    # noinspection PyShadowingNames
    def noise(
        self,
        x: ArrayLike,
        *,
        axis: int = -1,
        seed: int | None = None,
    ) -> np.ndarray:
        r"""
        Compute a time-domain noise array.

        Parameters
        ----------
        x :  array_like
            Time-domain signal.
        axis : int, optional
            Axis over which to apply the correction. If not given, applies over
            the last axis in ``x``.
        seed : int or None, optional
            Random number generator seed.

        Returns
        -------
        noise : ndarray
            Time-domain noise, in signal units.

        Notes
        -----
        For noise parameters :math:`\sigma_\alpha`, :math:`\sigma_\beta`,
        :math:`\sigma_\tau` and signal vector :math:`\boldsymbol{\mu}`, the
        :math:`k`-th element of the time-domain noise amplitude vector
        :math:`\boldsymbol{\sigma}` is given by [1]_

        .. math:: \sigma_k = \sqrt{\sigma_\alpha^2 + \sigma_\beta^2\mu_k^2 \
            + \sigma_\tau^2(\mathbf{D}\boldsymbol{\mu})_k^2},

        where :math:`\mathbf{D}` is the time-domain derivative operator.

        References
        ----------
        .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
            Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
            likelihood parameter estimation in terahertz time-domain
            spectroscopy," Opt. Express **29**, 4912-4926 (2021),
            `<https://doi.org/10.1364/OE.417724>`_.

        Examples
        --------
        The following example shows a noise sample
        :math:`\sigma_\mu(t_k)\epsilon(t_k)` for noise parameters
        :math:`\sigma_\alpha = 10^{-4}`, :math:`\sigma_\beta = 10^{-2}`,
        :math:`\sigma_\tau = 10^{-3}` and the simulated signal :math:`\mu(t)`.

        .. plot::
           :include-source: True

            >>> import matplotlib.pyplot as plt
            >>> import thztools as thz

            >>> n, dt, t0 = 256, 0.05, 2.5
            >>> mu, t = thz.wave(n, dt, t0)
            >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
            >>> noise_mod = thz.NoiseModel(alpha=alpha, beta=beta, tau=tau,
            ... dt=dt)
            >>> noise = noise_mod.noise(mu)

            >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
            >>> axs[0].plot(t, noise / beta)
            >>> axs[0].set_ylabel(r"$\sigma_\mu\epsilon/(\sigma_\beta\mu_0)$")
            >>> axs[1].plot(t, mu)
            >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
            >>> axs[1].set_xlabel("t (ps)")
            >>> plt.show()
        """
        x = np.asarray(x)
        axis = int(axis)
        if x.ndim > 1:
            if axis != -1:
                x = np.moveaxis(x, axis, -1)

        amp = self.amplitude(x)
        rng = default_rng(seed)
        noise = amp * rng.standard_normal(size=x.shape)
        if x.ndim > 1:
            if axis != -1:
                noise = np.moveaxis(noise, -1, axis)

        return noise


# noinspection PyShadowingNames
def transfer_out(
    tfun: Callable,
    x: ArrayLike,
    dt: float,
    *,
    fft_sign: bool = True,
    args: tuple = (),
) -> np.ndarray:
    r"""
    Apply a transfer function to a waveform.

    Parameters
    ----------
    tfun : callable
        Transfer function.

            ``tfun(omega, *args) -> np.ndarray``

        where ``omega`` is an array of angular frequencies and ``args`` is a
        tuple of the fixed parameters needed to completely specify
        the function. The units of ``omega`` must be the inverse of the units
        of ``dt``, such as radians/picosecond.
    x : array_like
        Data array.
    dt : float
        Sampling time, normally in picoseconds.
    fft_sign : bool, optional
        Complex exponential sign convention for harmonic time dependence.
    args : tuple, optional
        Extra arguments passed to the transfer function.

    Returns
    -------
    y : np.ndarray
        Result of applying the transfer function to ``x``.

    Notes
    -----
    The output waveform is computed by transforming :math:`x[n]` into the
    frequency domain, multiplying by the transfer function :math:`H[n]`,
    then transforming back into the time domain.

    .. math:: y[n] = \mathcal{F}^{-1}\{H[n] \mathcal{F}\{x[n]\}\}

    Examples
    --------

    .. plot::
       :include-source: True

        Apply a transfer function that rescales and shifts the input.

            .. math:: H(\omega) = a\exp(-i\omega\tau).

        Note that this form assumes the :math:`e^{+i\omega t}` representation
        of harmonic time dependence, which corresponds to the default setting
        ``fft_sign=True``.

        >>> import matplotlib.pyplot as plt
        >>> import thztools as thz

        >>> n, dt, t0 = 256, 0.05, 2.5
        >>> x, t = thz.wave(n, dt, t0)

        >>> def shiftscale(_w, _a, _tau):
        >>>     return _a * np.exp(-1j * _w * _tau)
        >>>
        >>> y = thz.transfer_out(shiftscale, x, dt=dt, fft_sign=True,
        ...                      args=(0.5, 1))

        >>> _, ax = plt.subplots()
        >>>
        >>> ax.plot(t, x, label='x')
        >>> ax.plot(t, y, label='y')
        >>>
        >>> ax.legend()
        >>> ax.set_xlabel('t (ps)')
        >>> ax.set_ylabel('Amplitude (arb. units)')
        >>>
        >>> plt.show()

        If the transfer function is expressed using the :math:`e^{-i\omega t}`
        representation that is more common in physics,

            .. math:: H(\omega) = a\exp(i\omega\tau),

        set ``fft_sign=False``.

        >>> def shiftscale_phys(_w, _a, _tau):
        >>>     return _a * np.exp(1j * _w * _tau)
        >>>
        >>> y_p = thz.transfer_out(shiftscale_phys, x, dt=dt, fft_sign=False,
        ...                        args=(0.5, 1))

        >>> _, ax = plt.subplots()
        >>>
        >>> ax.plot(t, x, label='x')
        >>> ax.plot(t, y_p, label='y')
        >>>
        >>> ax.legend()
        >>> ax.set_xlabel('t (ps)')
        >>> ax.set_ylabel('Amplitude (arb. units)')
        >>>
        >>> plt.show()
    """
    x = np.asarray(x)
    if x.ndim != 1:
        msg = "x must be a one-dimensional array"
        raise ValueError(msg)

    if not isinstance(args, tuple):
        args = (args,)

    n = x.size
    f_scaled = rfftfreq(n)
    w = 2 * np.pi * f_scaled / dt
    h = tfun(w, *args)
    if not fft_sign:
        h = np.conj(h)

    y = np.fft.irfft(np.fft.rfft(x) * h, n=n)

    return y


# noinspection PyShadowingNames
def wave(
    n: int,
    dt: float,
    t0: float,
    *,
    a: float = 1.0,
    taur: float = 0.3,
    tauc: float = 0.1,
    fwhm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Simulate a terahertz waveform.

    Parameters
    ----------

    n : int
        Number of samples.
    dt : float
        Sampling time, normally in picoseconds.
    t0 : float
        Pulse center, normally in picoseconds.
    a : float, optional
        Peak amplitude. The default is one.
    taur, tauc, fwhm : float, optional
        Current pulse rise time, current pulse decay time, and laser pulse
        FWHM, respectively. The defaults are 0.3 ps, 0.1 ps, and 0.05 ps,
        respectively, and assume that ``ts`` and ``t0`` are also given in
        picoseconds.

    Returns
    -------

    x : ndarray
        Signal array.
    t : ndarray
        Array of time samples.

    Notes
    -----
    This function uses a simplified model for terahertz generation from a
    photoconducting switch [1]_. The impulse response of the switch is a
    current pulse with an exponential rise time :math:`\tau_r` and an
    exponential decay (capture) time, :math:`\tau_c`,

    .. math:: I(t) \propto (1 - e^{-t/\tau_r})e^{-t/\tau_c},

    which is convolved with a Gaussian laser pulse with a full-width,
    half-maximum pulsewidth of ``fwhm``.

    References
    ----------
    .. [1] D. Grischkowsky and N. Katzenellenbogen, "Femtosecond Pulses of THz
        Radiation: Physics and Applications," in Picosecond Electronics and
        Optoelectronics, Technical Digest Series (Optica Publishing Group,
        1991), paper WA2,
        `<https://doi.org/10.1364/PEO.1991.WA2>`_.

    Examples
    --------
    The following example shows the simulated signal :math:`\mu(t)` normalized
    to its peak magnitude, :math:`\mu_0`.

    .. plot::
       :include-source: True

        >>> import matplotlib.pyplot as plt
        >>> import thztools as thz
        >>> n, dt, t0 = 256, 0.05, 2.5
        >>> mu, t = thz.wave(n, dt, t0)

        >>> _, ax = plt.subplots(layout="constrained")
        >>> ax.plot(t, mu)
        >>> ax.set_xlabel("t (ps)")
        >>> ax.set_ylabel(r"$\mu/\mu_0$")
        >>> plt.show()
    """
    taul = fwhm / np.sqrt(2 * np.log(2))

    f_scaled = rfftfreq(n)

    w = 2 * np.pi * f_scaled / dt
    ell = np.exp(-((w * taul) ** 2) / 2) / np.sqrt(2 * np.pi * taul**2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (ell * r) ** 2 * np.exp(1j * w * t0)

    t = dt * np.arange(n)

    x = irfft(np.conj(s), n=n)
    x = a * x / np.max(x)

    return x, t


# noinspection PyShadowingNames
def scaleshift(
    x: ArrayLike,
    *,
    a: ArrayLike | None = None,
    eta: ArrayLike | None = None,
    dt: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    r"""
    Rescale and shift waveforms.

    Parameters
    ----------
    x : array_like
        Data array.
    a : array_like, optional
        Scale array.
    eta : array_like, optional
        Shift array.
    dt : float, optional
        Sampling time. Default is 1.0.
    axis : int, optional
        Axis over which to apply the correction. If not given, applies over the
        last axis in ``x``.

    Returns
    -------
    x_adjusted : ndarray
        Adjusted data array.

    Examples
    --------
    The following example makes an array with 4 identical copies of the
    signal ``mu`` returned by ``thztools.thzgen``. It then uses
    ``thztools.scaleshift`` to rescale each copy by
    ``a = [1.0, 0.5, 0.25, 0.125]`` and shift it by
    ``eta = [0.0, 1.0, 2.0, 3.0]``.

    .. plot::
       :include-source: True

        >>> import matplotlib.pyplot as plt
        >>> import thztools as thz
        >>> n, dt, t0 = 256, 0.05, 2.5
        >>> mu, t = thz.wave(n, dt, t0)
        >>> m = 4
        >>> x = np.repeat(np.atleast_2d(mu), m, axis=0)
        >>> a = 0.5**np.arange(m)
        >>> eta = np.arange(m)
        >>> x_adj = thz.scaleshift(x, a=a, eta=eta, dt=dt)

        >>> _, ax = plt.subplots(layout="constrained")
        >>> ax.plot(t, x_adj.T, label=[f"{k=}" for k in range(4)])
        >>> ax.legend()
        >>> ax.set_xlabel("t (ps)")
        >>> ax.set_ylabel(r"$x_{\mathrm{adj}, k}$")
        >>> plt.show()
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

    f_scaled = rfftfreq(n)
    w = 2 * np.pi * f_scaled / dt
    phase = np.expand_dims(eta, axis=eta.ndim) * w

    x_adjusted = np.fft.irfft(
        np.fft.rfft(x) * np.exp(-1j * phase), n=n
    ) * np.expand_dims(a, axis=a.ndim)

    if x.ndim > 1:
        if axis != -1:
            x_adjusted = np.moveaxis(x_adjusted, -1, axis)

    return x_adjusted


def _costfuntls(
    fun: Callable,
    theta: ArrayLike,
    mu: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    sigma_x: ArrayLike,
    sigma_y: ArrayLike,
    dt: float = 1.0,
) -> np.ndarray:
    r"""Computes the residual vector for the total least squares cost function.

    Parameters
    ----------
    fun : callable
        Transfer function.

            ``fun(p, w, *args, **kwargs) -> np.ndarray``

        Assumes the :math:`+i\omega t` convention for harmonic time dependence.

    theta : array_like
        Input parameters for the function.

    mu : array_like
        Estimated input signal.

    x : array_like
        Measured input signal.

    y : array_like
        Measured output signal.

    sigma_x : array_like
        Noise vector of the input signal.

    sigma_y : array_like
        Noise vector of the output signal.

    dt : float, optional
        Sampling time. Default set to 1.0.

    Returns
    -------
    res : array_like


    """
    theta = np.asarray(theta)
    mu = np.asarray(mu)
    x = np.asarray(x)
    y = np.asarray(y)
    sigma_x = np.asarray(sigma_x)
    sigma_y = np.asarray(sigma_y)

    n = x.shape[-1]
    delta_norm = (x - mu) / sigma_x
    w = 2 * np.pi * rfftfreq(n) / dt
    h_f = fun(theta, w)

    eps_norm = (y - irfft(rfft(mu) * h_f, n=n)) / sigma_y

    res = np.concatenate((delta_norm, eps_norm))

    return res


def _tdnll_scaled(
    x: ArrayLike,
    logv: ArrayLike,
    delta: ArrayLike,
    alpha: ArrayLike,
    eta_on_dt: ArrayLike,
    *,
    fix_logv: bool,
    fix_delta: bool,
    fix_alpha: bool,
    fix_eta: bool,
    scale_logv: ArrayLike,
    scale_delta: ArrayLike,
    scale_alpha: ArrayLike,
    scale_eta: ArrayLike,
    scale_v: float,
) -> tuple[float, np.ndarray]:
    r"""
    Compute the scaled negative log-likelihood for the time-domain noise model.

    Computes the scaled negative log-likelihood function for obtaining the
    data matrix ``x`` given ``logv``, ``delta``, ``alpha``, ``eta``,
    ``scale_logv``, ``scale_delta``, ``scale_alpha``, ``scale_eta``, and
    ``scale_v``.

    Parameters
    ----------
    x : array_like
        Data matrix with shape (m, n), row-oriented.
    logv : array_like
        Array of three noise parameters.
    delta : array_like
        Signal deviation vector with shape (n,).
    alpha: array_like
        Amplitude deviation vector with shape (m - 1,).
    eta_on_dt : array_like
        Normalized delay deviation vector with shape (m - 1,), equal to
        ``eta``/``dt``.
    fix_logv : bool
        Exclude noise parameters from gradiate calculation when ``True``.
    fix_delta : bool
        Exclude signal deviation vector from gradiate calculation when
        ``True``.
    fix_alpha : bool
        Exclude amplitude deviation vector from gradiate calculation when
        ``True``.
    fix_eta : bool
        Exclude delay deviation vector from gradiate calculation when ``True``.
    scale_logv : array_like
        Array of three scale parameters for ``logv``.
    scale_delta : array_like
        Array of scale parameters for ``delta`` with shape (n,).
    scale_alpha : array_like
        Array of scale parameters for ``alpha`` with shape (m - 1,).
    scale_eta : array_like
        Array of scale parameters for ``eta`` with shape (m - 1,).
    scale_v : float
        Scale parameter for overall variance.

    Returns
    -------
    nll_scaled : float
        Scaled negative log-likelihood.
    gradnll_scaled : array_like
        Gradient of the negative scaled log-likelihood function with respect to
        free parameters.
    """
    x = np.asarray(x)
    logv = np.asarray(logv)
    delta = np.asarray(delta)
    alpha = np.asarray(alpha)
    eta_on_dt = np.asarray(eta_on_dt)

    scale_logv = np.asarray(scale_logv)
    scale_delta = np.asarray(scale_delta)
    scale_alpha = np.asarray(scale_alpha)
    scale_eta = np.asarray(scale_eta)

    m, n = x.shape

    # Compute scaled variance, mu, a, and eta
    v_scaled = np.exp(logv * scale_logv)
    mu = x[0, :] - delta * scale_delta
    a = np.insert(1.0 + alpha * scale_alpha, 0, 1.0)
    eta_on_dt = np.insert(eta_on_dt * scale_eta, 0, 0.0)

    # Compute frequency vector and Fourier coefficients of mu
    f_scaled = rfftfreq(n)
    w_scaled = 2 * np.pi * f_scaled
    mu_f = rfft(mu)

    exp_iweta = np.exp(1j * np.outer(eta_on_dt, w_scaled))
    zeta_f = ((np.conj(exp_iweta) * mu_f).T * a).T
    zeta = irfft(zeta_f, n=n)

    # Compute negative - log likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Alternative case: A, eta, or both are not set to defaults
    dzeta = irfft(1j * w_scaled * zeta_f, n=n)

    valpha = v_scaled[0]
    vbeta = v_scaled[1] * zeta**2
    vtau = v_scaled[2] * dzeta**2
    vtot_scaled = valpha + vbeta + vtau

    resnormsq_scaled = ressq / vtot_scaled
    nll_scaled = (
        scale_v * m * n * (np.log(2 * np.pi) + np.log(scale_v)) / 2
        + scale_v * np.sum(np.log(vtot_scaled)) / 2
        + np.sum(resnormsq_scaled) / 2
    )

    # Compute gradient
    gradnll_scaled = np.array([])
    if not (fix_logv & fix_delta & fix_alpha & fix_eta):
        reswt = res / vtot_scaled
        dvar = (scale_v * vtot_scaled - ressq) / vtot_scaled**2
        if not fix_logv:
            # Gradient wrt logv
            gradnll_scaled = np.append(
                gradnll_scaled,
                0.5 * np.sum(dvar) * v_scaled[0] * scale_logv[0],
            )
            gradnll_scaled = np.append(
                gradnll_scaled,
                0.5 * np.sum(zeta**2 * dvar) * v_scaled[1] * scale_logv[1],
            )
            gradnll_scaled = np.append(
                gradnll_scaled,
                0.5 * np.sum(dzeta**2 * dvar) * v_scaled[2] * scale_logv[2],
            )
        if not fix_delta:
            # Gradient wrt delta
            p = rfft(v_scaled[1] * dvar * zeta - reswt) - 1j * v_scaled[
                2
            ] * w_scaled * rfft(dvar * dzeta)
            gradnll_scaled = np.append(
                gradnll_scaled,
                -np.sum((irfft(exp_iweta * p, n=n).T * a).T, axis=0)
                * scale_delta,
            )
        if not fix_alpha:
            # Gradient wrt alpha
            term = (vtot_scaled - valpha) * dvar - reswt * zeta
            dnllda = np.sum(term, axis=1).T / a
            # Exclude first term, which is held fixed
            gradnll_scaled = np.append(
                gradnll_scaled, dnllda[1:] * scale_alpha
            )
        if not fix_eta:
            # Gradient wrt eta
            ddzeta = irfft(-(w_scaled**2) * zeta_f, n=n)
            dnlldeta = -np.sum(
                dvar
                * (zeta * dzeta * v_scaled[1] + dzeta * ddzeta * v_scaled[2])
                - reswt * dzeta,
                axis=1,
            )
            # Exclude first term, which is held fixed
            gradnll_scaled = np.append(
                gradnll_scaled, dnlldeta[1:] * scale_eta
            )

    return nll_scaled, gradnll_scaled


def tdnoisefit(
    x: ArrayLike,
    *,
    dt: float = 1.0,
    v0: ArrayLike | None = None,
    mu0: ArrayLike | None = None,
    a0: ArrayLike | None = None,
    eta0: ArrayLike | None = None,
    fix_v: bool = False,
    fix_mu: bool = False,
    fix_a: bool = True,
    fix_eta: bool = True,
) -> NoiseResult:
    r"""
    Estimate noise model parameters.

    Computes the noise parameters sigma and the underlying signal vector ``mu``
    for the data matrix ``x``, where the columns of ``x`` are each noisy
    measurements of ``mu``.

    Parameters
    ----------
    x : ndarray, shape(n, m)
        Data array with `m` waveforms, each composed of `n` points.
    dt : float, optional
        Sampling time. Default is 1.0.
    v0 : ndarray, shape (3,), optional
        Initial guess, noise model parameters with size (3,), expressed as
        variance amplitudes.
    mu0 : ndarray, shape(n,), optional
        Initial guess, signal vector with size (n,). Default is first column of
        ``x``.
    a0 : ndarray, shape(m,), optional
        Initial guess, amplitude vector with size (m,). Default is one for all
        entries.
    eta0 : ndarray, shape(m,), optional
        Initial guess, delay vector with size (m,). Default is zero for all
        entries.
    fix_v : bool, optional
        Fix noise variance parameters. Default is False.
    fix_mu : bool, optional
        Fix signal vector. Default is False.
    fix_a : bool, optional
        Fix amplitude vector. Default is True.
    fix_eta : bool, optional
        Fix delay vector. Default is True.

    Returns
    -------
    res : NoiseResult
        Fit result, represented as a ``NoiseResult`` object.
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
        v0 = np.mean(np.var(x, 1)) * np.ones(NUM_NOISE_PARAMETERS)
    else:
        v0 = np.asarray(v0)
        if v0.size != NUM_NOISE_PARAMETERS:
            msg = (
                "Noise parameter array logv must have "
                f"{NUM_NOISE_PARAMETERS} elements."
            )
            raise ValueError(msg)

    if mu0 is None:
        mu0 = x[:, 0]
    else:
        mu0 = np.asarray(mu0)
        if mu0.size != n:
            msg = "Size of mu0 is incompatible with data array x."
            raise ValueError(msg)

    scale_logv = 1e0 * np.ones(3)
    alpha, beta, tau = np.sqrt(v0)
    noise_model = NoiseModel(alpha, beta, tau, dt)
    scale_delta = 1e-0 * noise_model.amplitude(x[:, 0])
    scale_alpha = 1e-2 * np.ones(m - 1)
    scale_eta = 1e-3 * np.ones(m - 1)
    scale_v = 1.0e-6

    # Replace log(x) with -inf when x <= 0
    v0_scaled = np.asarray(v0, dtype=float) / scale_v
    logv0_scaled = np.ma.log(v0_scaled).filled(-np.inf) / scale_logv
    delta0 = (x[:, 0] - mu0) / scale_delta

    if a0 is None:
        a0 = np.ones(m)
    else:
        a0 = np.asarray(a0)
        if a0.size != m:
            msg = "Size of a0 is incompatible with data array x."
            raise ValueError(msg)

    alpha0 = (a0[1:] - 1.0) / (a0[0] * scale_alpha)

    if eta0 is None:
        eta0 = np.zeros(m)
    else:
        eta0 = np.asarray(eta0)
        if eta0.size != m:
            msg = "Size of eta0 is incompatible with data array x."
            raise ValueError(msg)

    eta0 = eta0[1:] / scale_eta

    # Set initial guesses for all free parameters
    x0 = np.array([])
    if not fix_v:
        x0 = np.concatenate((x0, logv0_scaled))
    if not fix_mu:
        x0 = np.concatenate((x0, delta0))
    if not fix_a:
        x0 = np.concatenate((x0, alpha0))
    if not fix_eta:
        x0 = np.concatenate((x0, eta0))

    # Bundle free parameters together into objective function
    def objective(_p):
        if fix_v:
            _logv = logv0_scaled
        else:
            _logv = _p[:3]
            _p = _p[3:]
        if fix_mu:
            _delta = delta0
        else:
            _delta = _p[:n]
            _p = _p[n:]
        if fix_a:
            _alpha = alpha0
        else:
            _alpha = _p[: m - 1]
            _p = _p[m - 1 :]
        if fix_eta:
            _eta = eta0
        else:
            _eta = _p[: m - 1]
        return _tdnll_scaled(
            x.T,
            _logv,
            _delta,
            _alpha,
            _eta / dt,
            fix_logv=fix_v,
            fix_delta=fix_mu,
            fix_alpha=fix_a,
            fix_eta=fix_eta,
            scale_logv=scale_logv,
            scale_delta=scale_delta,
            scale_alpha=scale_alpha,
            scale_eta=scale_eta,
            scale_v=scale_v,
        )

    # Minimize cost function with respect to free parameters
    out = minimize(
        objective,
        x0,
        method="BFGS",
        jac=True,
    )

    # Parse output
    p = {}
    x_out = out.x
    if fix_v:
        p["var"] = v0_scaled * scale_v
    else:
        p["var"] = np.exp(x_out[:3] * scale_logv) * scale_v
        x_out = x_out[3:]

    if fix_mu:
        p["mu"] = mu0
    else:
        p["mu"] = x[:, 0] - x_out[:n] * scale_delta
        x_out = x_out[n:]

    if fix_a:
        p["a"] = a0
    else:
        p["a"] = np.concatenate(([1.0], 1.0 + x_out[: m - 1] * scale_alpha))
        x_out = x_out[m - 1 :]

    if fix_eta:
        p["eta"] = eta0
    else:
        p["eta"] = np.concatenate(([0.0], x_out[: m - 1] * scale_eta))

    diagnostic = {
        "grad_scaled": out.jac,
        "hess_inv_scaled": out.hess_inv,
        "err": {
            "var": np.array([]),
            "delta": np.array([]),
            "a": np.array([]),
            "eta": np.array([]),
        },
        "success": out.success,
        "status": out.status,
        "message": out.message,
        "nfev": out.nfev,
        "njev": out.njev,
        "nit": out.nit,
    }

    # Concatenate scaling vectors for all sets of free parameters
    scale_hess_inv = np.concatenate(
        [
            val
            for tf, val in zip(
                [fix_v, fix_mu, fix_a, fix_eta],
                [scale_logv, scale_delta, scale_alpha, scale_eta],
            )
            if not tf
        ]
    )

    # Convert inverse Hessian into unscaled parameters
    hess_inv = scale_v * (
        np.diag(scale_hess_inv)
        @ diagnostic["hess_inv_scaled"]
        @ np.diag(scale_hess_inv)
    )

    # Determine parameter uncertainty vector from diagonal entries
    err = np.sqrt(np.diag(hess_inv))

    # Parse error vector
    if not fix_v:
        # Propagate error from log(V) to V
        diagnostic["err"]["var"] = np.sqrt(
            np.diag(np.diag(p["var"]) @ hess_inv[0:3, 0:3]) @ np.diag(p["var"])
        )
        err = err[3:]

    if not fix_mu:
        diagnostic["err"]["delta"] = err[:n]
        err = err[n:]

    if not fix_a:
        diagnostic["err"]["alpha"] = np.concatenate(([0], err[: m - 1]))
        err = err[m - 1 :]

    if not fix_eta:
        diagnostic["err"]["eta"] = np.concatenate(([0], err[: m - 1]))

    return NoiseResult(p, out.fun / scale_v, diagnostic)


def fit(
    fun: Callable,
    p0: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    dt: float = 1.0,
    sigma_parms: ArrayLike = (1.0, 0.0, 0.0),
    f_bounds: ArrayLike = (0.0, np.inf),
    p_bounds: ArrayLike | None = None,
    jac: Callable | None = None,
    args: ArrayLike = (),
    kwargs: dict | None = None,
) -> dict:
    r"""
    Fit a transfer function to time-domain data.

    Computes the noise on the input ``x`` and output ``y`` time series using
    a given ``NoiseModel``. Uses the total residuals generated by
    ``_costfuntls`` to fit the input and output to the transfer function.

    Parameters
    ----------
    fun : callable
        Transfer function.

            ``fun(p, w, *args, **kwargs) -> np.ndarray``

        Assumes the :math:`+i\omega t` convention for harmonic time dependence.
    p0 : array_like
        Initial guess for ``p``.
    x : array_like
        Measured input signal.
    y : array_like
        Measured output signal.
    dt : float, optional
        Sampling time.
    sigma_parms : None or array_like, optional
        Noise parameters with size (3,), expressed as standard deviation
        amplitudes.
    f_bounds : array_like, optional
        Frequency bounds.
    p_bounds : None, 2-tuple of array_like, or Bounds, optional
        Lower and upper bounds on fit parameter(s).
    jac : None or callable, optional
        Method of calculating derivative of the output signal residuals
        with respect to the fit parameter(s), theta.
    args : tuple, optional
        Additional arguments passed to ``fun`` and ``jac``.
    kwargs : dict, optional
        Additional keyword arguments passed to ``fun`` and ``jac``.

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
                The value of chi-squared.
            delta : array_like
                Residuals of the input waveform ``x``.
            epsilon : array_like
                Resiuals of the output waveform ``y``.
            success : bool
                True if one of the convergence criteria is satisfied.
    """
    fit_method = "trf"

    p0 = np.asarray(p0)
    x = np.asarray(x)
    y = np.asarray(y)
    sigma_parms = np.asarray(sigma_parms)

    n = y.shape[-1]
    n_p = len(p0)

    if p_bounds is None:
        p_bounds = (-np.inf, np.inf)
        fit_method = "lm"
    else:
        p_bounds = np.asarray(p_bounds)
        if p_bounds.shape[0] == 2:  # noqa: PLR2004
            p_bounds = (
                np.concatenate((p_bounds[0], np.full((n,), -np.inf))),
                np.concatenate((p_bounds[1], np.full((n,), np.inf))),
            )
        else:
            msg = "`bounds` must contain 2 elements."
            raise ValueError(msg)

    if kwargs is None:
        kwargs = {}

    w = 2 * np.pi * rfftfreq(n, dt)
    w_bounds = 2 * np.pi * np.asarray(f_bounds)
    w_below_idx = w <= w_bounds[0]
    w_above_idx = w > w_bounds[1]
    w_in_idx = np.invert(w_below_idx) * np.invert(w_above_idx)
    n_f = len(w)

    alpha, beta, tau = sigma_parms
    noise_model = NoiseModel(alpha, beta, tau, dt)
    sigma_x = noise_model.amplitude(x)
    sigma_y = noise_model.amplitude(y)
    p0_est = np.concatenate((p0, np.zeros(n)))

    def etfe(_x, _y):
        return rfft(_y) / rfft(_x)

    def function(_theta, _w):
        _h = etfe(x, y)
        _w_in = _w[w_in_idx]
        return np.concatenate(
            (
                _h[w_below_idx],
                fun(_theta, _w_in, *args, **kwargs),
                _h[w_above_idx],
            )
        )

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
        mu_est = x[:] - _x[n_p:]
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
        lambda _p: _costfuntls(
            function,
            _p[:n_p],
            x[:] - _p[n_p:],
            x[:],
            y[:],
            sigma_x,
            sigma_y,
            dt,
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
    p["mu_opt"] = x - result.x[n_p:]
    _, s, vt = la.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    p["p_var"] = np.diag(np.dot(vt.T / s**2, vt))[:n_p]
    p["mu_var"] = np.diag(np.dot(vt.T / s**2, vt))[n_p:]
    p["resnorm"] = 2 * result.cost
    p["delta"] = x - p["mu_opt"]
    p["epsilon"] = y - irfft(rfft(p["mu_opt"]) * function(p["p_opt"], w), n=n)
    p["success"] = result.success
    return p
