from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from numpy.fft import irfft, rfft, rfftfreq
from numpy.random import default_rng
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize import approx_fprime as fprime

NUM_NOISE_PARAMETERS = 3
NUM_NOISE_DATA_DIMENSIONS = 2


@dataclass
class GlobalOptions:
    r"""
    Class for global options.

    Attributes
    ----------
    sampling_time : float | None, optional
        Global sampling time, normally in picoseconds. When set to None, the
        default, times and frequencies are treated as dimensionless quantities
        that are scaled by the (undetermined) sampling time.
    """
    sampling_time: float | None


options = GlobalOptions


def set_option(key: str, value: Any) -> None:
    r"""
    Set global option.

    Parameters
    ----------
    key : str
        Option name.
    value : object
        Option value.

    Returns
    -------
    None
        No return value.
    """
    setattr(options, key, value)


def get_option(key: str) -> Any:
    r"""
    Get global option.

    Parameters
    ----------
    key : str
        Option name.

    Returns
    -------
    val : object
        Option value.
    """
    return getattr(options, key)


def _assign_sampling_time(dt: float | None) -> float:
    dt_out = 1.0
    if dt is None and get_option("sampling_time") is not None:
        dt_out = get_option("sampling_time")
    elif dt is not None and get_option("sampling_time") is None:
        dt_out = dt
    elif dt is not None and get_option("sampling_time") is not None:
        if np.isclose(dt, get_option("sampling_time")):
            dt_out = dt
        else:
            opt_sampling_time = get_option("sampling_time")
            msg = (
                f"Input sampling time {dt=} conflicts with "
                f"{opt_sampling_time=}, using {dt=}"
            )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            dt_out = dt
    return dt_out


@dataclass
class NoiseModel:
    r"""
    Noise model class.

    For noise parameters :math:`\sigma_\alpha`, :math:`\sigma_\beta`,
    :math:`\sigma_\tau` and signal vector :math:`\boldsymbol{\mu}`, the
    :math:`k`-th element of the time-domain noise variance vector
    :math:`\boldsymbol{\sigma}^2` is given by [1]_

    .. math:: \sigma_k^2 = \sigma_\alpha^2 + \sigma_\beta^2\mu_k^2 \
        + \sigma_\tau^2(\mathbf{D}\boldsymbol{\mu})_k^2,

    where :math:`\mathbf{D}` is the time-domain derivative operator.

    Parameters
    ----------
    sigma_alpha : float
        Additive noise amplitude.
    sigma_beta : float
        Multiplicative noise amplitude.
    sigma_tau : float
        Timebase noise amplitude.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        :attr:`dt` and ``thztools.get_option("sampling_time")`` are ``None``,
        the :attr:`sigma_tau` parameter is given in units of the sampling time.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the :attr:`dt`
        parameter are both not ``None`` and are set to different ``float``
        values, the function will set the sampling time to :attr:`dt` and raise
        a :class:`UserWarning`.

    See Also
    --------
    noisefit : Estimate noise model parameters.

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

    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> thz.set_option("sampling_time", dt)
    >>> t = thz.timebase(n)
    >>> mu = thz.wave(n)
    >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
    >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
    ...  sigma_tau=tau)
    >>> noise_variance = noise_model.noise_var(mu)

    >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
    >>> axs[0].plot(t, noise_variance / beta**2)
    >>> axs[0].set_ylabel(r"$\sigma^2/(\sigma_\beta\mu_0)^2$")
    >>> axs[1].plot(t, mu)
    >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
    >>> axs[1].set_xlabel("t (ps)")
    >>> plt.show()
    """
    sigma_alpha: float
    sigma_beta: float
    sigma_tau: float
    dt: float | None = None

    # noinspection PyShadowingNames
    def noise_var(
        self, x: ArrayLike, *, axis: int = -1
    ) -> NDArray[np.float64]:
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
        noise_variance : ndarray
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

        >>> import thztools as thz
        >>> from matplotlib import pyplot as plt
        >>> n, dt = 256, 0.05
        >>> thz.set_option("sampling_time", dt)
        >>> t = thz.timebase(n)
        >>> mu = thz.wave(n)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau)
        >>> noise_variance = noise_model.noise_var(mu)

        >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        >>> axs[0].plot(t, noise_variance / beta**2)
        >>> axs[0].set_ylabel(r"$\sigma^2/(\sigma_\beta\mu_0)^2$")
        >>> axs[1].plot(t, mu)
        >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
        >>> axs[1].set_xlabel("t (ps)")
        >>> plt.show()
        """
        dt = _assign_sampling_time(self.dt)
        x = np.asarray(x, dtype=np.float64)
        axis = int(axis)
        if x.ndim > 1:
            if axis != -1:
                x = np.moveaxis(x, axis, -1)

        n = x.shape[-1]
        w_scaled = 2 * np.pi * rfftfreq(n)
        xdot = irfft(1j * w_scaled * rfft(x), n=n) / dt

        noise_variance = (
            self.sigma_alpha**2
            + (self.sigma_beta * x) ** 2
            + (self.sigma_tau * xdot) ** 2
        )

        if x.ndim > 1:
            if axis != -1:
                noise_variance = np.moveaxis(noise_variance, -1, axis)

        return noise_variance

    # noinspection PyShadowingNames
    def noise_amp(
        self, x: ArrayLike, *, axis: int = -1
    ) -> NDArray[np.float64]:
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
        noise_amplitude : ndarray
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

        >>> import thztools as thz
        >>> from matplotlib import pyplot as plt
        >>> n, dt = 256, 0.05
        >>> thz.set_option("sampling_time", dt)
        >>> t = thz.timebase(n)
        >>> mu = thz.wave(n)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau)
        >>> noise_amplitude = noise_model.noise_amp(mu)

        >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        >>> axs[0].plot(t, noise_amplitude / beta)
        >>> axs[0].set_ylabel(r"$\sigma/(\sigma_\beta\mu_0)$")
        >>> axs[1].plot(t, mu)
        >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
        >>> axs[1].set_xlabel("t (ps)")
        >>> plt.show()
        """
        return np.sqrt(self.noise_var(x, axis=axis))

    # noinspection PyShadowingNames
    def noise_sim(
        self,
        x: ArrayLike,
        *,
        axis: int = -1,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        r"""
        Simulate time-domain noise.

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
        noise_simulation : ndarray
            Simulated time-domain noise, in signal units.

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

        >>> import thztools as thz
        >>> from matplotlib import pyplot as plt
        >>> n, dt = 256, 0.05
        >>> thz.set_option("sampling_time", dt)
        >>> t = thz.timebase(n)
        >>> mu = thz.wave(n)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau)
        >>> noise = noise_model.noise_sim(mu, seed=1234)

        >>> _, axs = plt.subplots(2, 1, sharex=True, layout="constrained")
        >>> axs[0].plot(t, noise / beta)
        >>> axs[0].set_ylabel(r"$\sigma_\mu\epsilon/(\sigma_\beta\mu_0)$")
        >>> axs[1].plot(t, mu)
        >>> axs[1].set_ylabel(r"$\mu/\mu_0$")
        >>> axs[1].set_xlabel("t (ps)")
        >>> plt.show()
        """
        x = np.asarray(x, dtype=np.float64)
        axis = int(axis)
        if x.ndim > 1:
            if axis != -1:
                x = np.moveaxis(x, axis, -1)

        amp = self.noise_amp(x)
        rng = default_rng(seed)
        noise = amp * rng.standard_normal(size=x.shape)
        if x.ndim > 1:
            if axis != -1:
                noise = np.moveaxis(noise, -1, axis)

        return noise


# noinspection PyShadowingNames
@dataclass
class NoiseResult:
    r"""
    Dataclass for the output of :func:`noisefit`.

    Parameters
    ----------
    noise_model : NoiseModel
        Noise parameters, represented as a :class:`NoiseModel` object.
    mu : ndarray, shape (n,)
        Signal vector.
    a : ndarray, shape (m,)
        Signal amplitude drift vector.
    eta : ndarray, shape (m,)
        Signal delay drift vector.
    fval : float
        Value of optimized NLL cost function.
    hess_inv : ndarray
        Inverse Hessian matrix of optimized NLL cost function.
    err_sigma_alpha, err_sigma_beta, err_sigma_tau : float
        Estimated uncertainty in the noise model parameters. Set equal to 0.0
        when the parameter is fixed.
    err_mu : ndarray
        Estimated uncertainty in ``mu``.
    err_a : ndarray
        Estimated uncertainty in ``a``.
    err_eta : ndarray
        Estimated uncertainty in ``eta``.
    diagnostic : scipy.optimize.OptimizeResult
        Instance of :class:`scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.minimize`. Note that the attributes ``fun``,
        ``jac``, and ``hess_inv`` represent functions over the internally
        scaled parameters.

    See Also
    --------
    NoiseModel : Noise model class.
    noisefit : Estimate noise model parameters.
    """
    noise_model: NoiseModel
    mu: NDArray[np.float64]
    a: NDArray[np.float64]
    eta: NDArray[np.float64]
    fval: float
    hess_inv: NDArray[np.float64]
    err_sigma_alpha: float
    err_sigma_beta: float
    err_sigma_tau: float
    err_mu: NDArray[np.float64]
    err_a: NDArray[np.float64]
    err_eta: NDArray[np.float64]
    diagnostic: OptimizeResult


# noinspection PyShadowingNames
def transfer(
    tfun: Callable,
    x: ArrayLike,
    *,
    dt: float | None = None,
    numpy_sign_convention: bool = True,
    args: tuple = (),
) -> NDArray[np.float64]:
    r"""
    Apply a transfer function to a waveform.

    Parameters
    ----------
    tfun : callable
        Transfer function.

            ``tfun(omega, *args) -> ndarray``

        where ``omega`` is an array of angular frequencies and ``args`` is a
        tuple of the fixed parameters needed to completely specify
        the function. The units of ``omega`` must be the inverse of the units
        of ``dt``, such as radians/picosecond.
    x : array_like
        Data array.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to ``1.0``. In this case, the angular frequency
        ``omega`` must be given in units of radians per sampling time, and any
        parameters in ``args`` must be expressed with the sampling time as the
        unit of time.
    numpy_sign_convention : bool, optional
        Adopt NumPy sign convention for harmonic time dependence, e.g, express
        a harmonic function with frequency :math:`\omega` as
        :math:`x(t) = a e^{i\omega t}`. Default is ``True``. Uses the
        convention more common in physics, :math:`x(t) = a e^{-i\omega t}`,
        when set to ``False``.
    args : tuple, optional
        Extra arguments passed to the transfer function.

    Returns
    -------
    y : ndarray
        Result of applying the transfer function to ``x``.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

    See Also
    --------
    fit : Fit a transfer function to time-domain data.

    Notes
    -----
    The output waveform is computed by transforming :math:`x[n]` into the
    frequency domain, multiplying by the transfer function :math:`H[n]`,
    then transforming back into the time domain.

    .. math:: y[n] = \mathcal{F}^{-1}\{H[n] \mathcal{F}\{x[n]\}\}

    Examples
    --------
    Apply a transfer function that rescales the input by :math:`a` and shifts
    it by :math:`\tau`.

    .. math:: H(\omega) = a\exp(-i\omega\tau).

    Note that this form assumes the :math:`e^{+i\omega t}` representation
    of harmonic time dependence, which corresponds to the default setting
    ``fft_sign=True``.

    If the transfer function is expressed using the :math:`e^{-i\omega t}`
    representation, more common in physics,

    .. math:: H(\omega) = a\exp(i\omega\tau),

    set ``fft_sign=False``.

    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> thz.set_option("sampling_time", dt)
    >>> t = thz.timebase(n)
    >>> x = thz.wave(n)
    >>> def shiftscale(_w, _a, _tau):
    ...     return _a * np.exp(-1j * _w * _tau)
    >>>
    >>> y = thz.transfer(shiftscale, x,
    ...                  numpy_sign_convention=True,
    ...                  args=(0.5, 1))

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

    >>> def shiftscale_phys(_w, _a, _tau):
    ...     return _a * np.exp(1j * _w * _tau)
    >>>
    >>> y_p = thz.transfer(shiftscale_phys, x,
    ...                    numpy_sign_convention=False,
    ...                    args=(0.5, 1))

    >>> _, ax = plt.subplots()
    >>>
    >>> ax.plot(t, x, label='x');
    >>> ax.plot(t, y_p, label='y');
    >>>
    >>> ax.legend();
    >>> ax.set_xlabel('t (ps)');
    >>> ax.set_ylabel('Amplitude (arb. units)');
    >>>
    >>> plt.show()
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        msg = "x must be a one-dimensional array"
        raise ValueError(msg)

    if not isinstance(args, tuple):
        args = (args,)

    dt = _assign_sampling_time(dt)
    n = x.size
    w_scaled = 2 * np.pi * rfftfreq(n)
    h = tfun(w_scaled / dt, *args)
    if not numpy_sign_convention:
        h = np.conj(h)

    y = np.fft.irfft(np.fft.rfft(x) * h, n=n)

    return y


# noinspection PyShadowingNames
def timebase(
    n: int, *, dt: float | None = None, t_init: float = 0.0
) -> NDArray[np.float64]:
    r"""
    Timebase for time-domain waveforms.

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to 1.0.
    t_init : float, optional
        Value of the initial time point. Default is ``0.0``.

    Returns
    -------
    t : ndarray
        Array of time samples.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

    Notes
    -----
    This is a helper function that computes

        ``t = t_init + dt * numpy.arange(n)``.

    Examples
    --------
    The following example computes the timebase with different methods for
    assigning the sampling time.

    >>> import thztools as thz
    >>> n, dt, t_init = 256, 0.05, 2.5
    >>> t_1 = thz.timebase(n)
    >>> t_2 = thz.timebase(n, dt=dt)
    >>> thz.set_option("sampling_time", dt)
    >>> t_3 = thz.timebase(n)
    >>> t_4 = thz.timebase(n, t_init=t_init)
    >>> print(t_1[:3])
    [0. 1. 2.]
    >>> print(t_2[:3])
    [0.   0.05 0.1 ]
    >>> print(t_3[:3])
    [0.   0.05 0.1 ]
    >>> print(t_4[:3])
    [2.5  2.55 2.6 ]
    """
    dt = _assign_sampling_time(dt)
    return t_init + dt * np.arange(n)


def wave(
    n: int,
    *,
    dt: float | None = None,
    t0: float | None = None,
    a: float = 1.0,
    taur: float | None = None,
    tauc: float | None = None,
    fwhm: float | None = None,
) -> NDArray[np.float64]:
    r"""
    Simulate a terahertz waveform.

    Parameters
    ----------

    n : int
        Number of samples.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to 1.0.
    t0 : float or None, optional
        Pulse location, normally in picoseconds. Default is ``0.3 * n * dt``.
    a : float, optional
        Peak amplitude. The default is one.
    taur, tauc, fwhm : float or None, optional
        Current pulse rise time, current pulse decay time, and laser pulse
        FWHM, respectively. The defaults are ``6.0 * dt``, ``2.0 * dt``, and
        ``1.0 * dt``, respectively.

    Returns
    -------

    x : ndarray
        Signal array.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

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

    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> thz.set_option("sampling_time", dt)
    >>> t = thz.timebase(n)
    >>> mu = thz.wave(n)

    >>> _, ax = plt.subplots(layout="constrained")
    >>> ax.plot(t, mu);
    >>> ax.set_xlabel("t (ps)");
    >>> ax.set_ylabel(r"$\mu/\mu_0$");
    >>> plt.show()
    """
    dt = _assign_sampling_time(dt)
    if t0 is None:
        t0 = 0.3 * n * dt

    if taur is None:
        taur = 6.0 * dt

    if tauc is None:
        tauc = 2.0 * dt

    if fwhm is None:
        fwhm = dt

    taul = fwhm / np.sqrt(2 * np.log(2))

    f_scaled = rfftfreq(n)

    w = 2 * np.pi * f_scaled / dt
    ell = np.exp(-((w * taul) ** 2) / 2) / np.sqrt(2 * np.pi * taul**2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (ell * r) ** 2 * np.exp(1j * w * t0)

    x = irfft(np.conj(s), n=n)
    x = a * x / np.max(x)

    return x


# noinspection PyShadowingNames
def scaleshift(
    x: ArrayLike,
    *,
    dt: float | None = None,
    a: ArrayLike | None = None,
    eta: ArrayLike | None = None,
    axis: int = -1,
) -> NDArray[np.float64]:
    r"""
    Rescale and shift waveforms.

    Parameters
    ----------
    x : array_like
        Data array.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to ``1.0``. In this case, ``eta`` must be given in
        units of the sampling time.
    a : array_like, optional
        Scale array.
    eta : array_like, optional
        Shift array.
    axis : int, optional
        Axis over which to apply the correction. If not given, applies over the
        last axis in ``x``.

    Returns
    -------
    x_adj : ndarray
        Adjusted data array.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

    Examples
    --------
    The following example makes an array with 4 identical copies of the
    signal ``mu`` returned by :func:`wave`. It then uses :func:`scaleshift` to
    rescale each copy by ``a = [1.0, 0.5, 0.25, 0.125]`` and shift it by
    ``eta = [0.0, 1.0, 2.0, 3.0]``.

    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> thz.set_option("sampling_time", dt)
    >>> t = thz.timebase(n)
    >>> mu = thz.wave(n)
    >>> m = 4
    >>> x = np.repeat(np.atleast_2d(mu), m, axis=0)
    >>> a = 0.5**np.arange(m)
    >>> eta = np.arange(m)
    >>> x_adj = thz.scaleshift(x, a=a, eta=eta, dt=dt)

    >>> plt.plot(t, x_adj.T, label=[f"{k=}" for k in range(4)]);
    >>> plt.legend();
    >>> plt.xlabel("t (ps)");
    >>> plt.ylabel(r"$x_{\mathrm{adj}, k}$");
    >>> plt.show()
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.empty(x.shape)

    dt = _assign_sampling_time(dt)

    axis = int(axis)
    if x.ndim > 1:
        if axis != -1:
            x = np.moveaxis(x, axis, -1)

    n = x.shape[-1]
    m = x.shape[:-1]

    if a is None:
        a = np.ones(m)
    else:
        a = np.asarray(a, dtype=np.float64)
        if a.shape != m:
            msg = (
                f"Scale correction with shape {a.shape} can not be applied "
                f"to data with shape {x.shape}"
            )
            raise ValueError(msg)

    if eta is None:
        eta = np.zeros(m)
    else:
        eta = np.asarray(eta, dtype=np.float64)
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
    dt: float | None = 1.0,
) -> NDArray[np.float64]:
    r"""Computes the residual vector for the total least squares cost function.

    Parameters
    ----------
    fun : callable
        Transfer function.

            ``fun(p, w, *args, **kwargs) -> ndarray``

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
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to ``1.0``. In this case, the angular frequency
        ``omega`` must be given in units of radians per sampling time, and any
        parameters in ``args`` must be expressed with the sampling time as the
        unit of time.

    Returns
    -------
    res : ndarray
        Residual array.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.
    """
    theta = np.asarray(theta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma_x = np.asarray(sigma_x, dtype=np.float64)
    sigma_y = np.asarray(sigma_y, dtype=np.float64)

    dt = _assign_sampling_time(dt)

    n = x.shape[-1]
    delta_norm = (x - mu) / sigma_x
    w = 2 * np.pi * rfftfreq(n) / dt
    h_f = fun(theta, w)

    eps_norm = (y - irfft(rfft(mu) * h_f, n=n)) / sigma_y

    res = np.concatenate((delta_norm, eps_norm))

    return res


def _costfun_noisefit(
    x: NDArray[np.float64],
    logv_alpha: float,
    logv_beta: float,
    logv_tau: float,
    delta_mu: NDArray[np.float64],
    delta_a: NDArray[np.float64],
    eta: NDArray[np.float64],
    *,
    fix_logv_alpha: bool,
    fix_logv_beta: bool,
    fix_logv_tau: bool,
    fix_delta_mu: bool,
    fix_delta_a: bool,
    fix_eta: bool,
    scale_sigma_alpha: float,
    scale_sigma_beta: float,
    scale_sigma_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    r"""
    Compute the scaled cost function for the time-domain noise model.

    Computes the scaled maximum-likelihood cost function for obtaining the
    data matrix ``x`` given ``logv``, ``delta``, ``alpha``, ``eta``,
    ``scale_sigma_alpha``, ``scale_sigma_beta``, ``scale_sigma_tau``,
    ``scale_delta_mu``, ``scale_delta_a``, and ``scale_eta``.

    Parameters
    ----------
    x : ndarray
        Data matrix with shape (m, n), row-oriented.
    logv_alpha, logv_beta, logv_tau : float
        Logarithm of the associated noise variance parameter, with sigma_tau
        given in units of sampling time.
    delta_mu : ndarray
        Signal deviation vector with shape (n,).
    delta_a: ndarray
        Amplitude deviation vector with shape (m - 1,).
    eta : ndarray
        Delay deviation vector with shape (m - 1,), given in units of sampling
        time.
    fix_logv_alpha, fix_logv_beta, fix_logv_tau : bool
        Exclude noise parameter from gradiate calculation when ``True``.
    fix_delta_mu : bool
        Exclude signal deviation vector from gradiate calculation when
        ``True``.
    fix_delta_a : bool
        Exclude signal amplitude deviation vector from gradiate calculation
        when ``True``.
    fix_eta : bool
        Exclude signal delay deviation vector from gradiate calculation when
        ``True``.
    scale_sigma_alpha : float
        Scale parameter for ``sigma_alpha``.
    scale_sigma_beta : float
        Scale parameter for ``sigma_beta``.
    scale_sigma_tau : float
        Scale parameter for ``sigma_tau``.
    scale_delta_mu : ndarray
        Array of scale parameters for ``delta`` with shape (n,).
    scale_delta_a : ndarray
        Array of scale parameters for ``alpha`` with shape (m - 1,).
    scale_eta : ndarray
        Array of scale parameters for ``eta`` with shape (m - 1,).

    Returns
    -------
    costfun_scaled : float
        Scaled negative log-likelihood.
    gradnll_scaled : ndarray
        Gradient of the negative scaled log-likelihood function with respect to
        free parameters.
    """
    m, n = x.shape

    logv = np.asarray([logv_alpha, logv_beta, logv_tau], dtype=np.float64)

    # Compute scaled noise model parameters, mu, a, and eta
    scale_sigma = np.array(
        [scale_sigma_alpha, scale_sigma_beta, scale_sigma_tau],
        dtype=np.float64,
    )
    scale_v = scale_sigma**2
    v_scaled = np.exp(logv) * scale_v
    mu = x[0, :] - delta_mu * scale_delta_mu
    a = np.insert(1.0 + delta_a * scale_delta_a, 0, 1.0)
    eta = np.insert(eta * scale_eta, 0, 0.0)

    # Compute frequency vector and Fourier coefficients of mu
    f_scaled = rfftfreq(n)
    w_scaled = 2 * np.pi * f_scaled
    mu_f = rfft(mu)

    exp_iweta = np.exp(1j * np.outer(eta, w_scaled))
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
    vtot = valpha + vbeta + vtau

    resnormsq_scaled = ressq / vtot
    costfun_scaled = np.sum(np.log(vtot)) + np.sum(resnormsq_scaled)

    # Compute gradient
    gradnll_scaled = np.array([], dtype=np.float64)
    if not (
        fix_logv_alpha
        & fix_logv_beta
        & fix_logv_tau
        & fix_delta_mu
        & fix_delta_a
        & fix_eta
    ):
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot**2
        # Gradient wrt logv
        if not fix_logv_alpha:
            gradnll_scaled = np.append(
                gradnll_scaled,
                np.sum(dvar) * v_scaled[0],
            )
        if not fix_logv_beta:
            gradnll_scaled = np.append(
                gradnll_scaled,
                np.sum(zeta**2 * dvar) * v_scaled[1],
            )
        if not fix_logv_tau:
            gradnll_scaled = np.append(
                gradnll_scaled,
                np.sum(dzeta**2 * dvar) * v_scaled[2],
            )
        if not fix_delta_mu:
            # Gradient wrt delta
            p = rfft(v_scaled[1] * dvar * zeta - reswt) - 1j * v_scaled[
                2
            ] * w_scaled * rfft(dvar * dzeta)
            gradnll_scaled = np.append(
                gradnll_scaled,
                -2
                * np.sum((irfft(exp_iweta * p, n=n).T * a).T, axis=0)
                * scale_delta_mu,
            )
        if not fix_delta_a:
            # Gradient wrt alpha
            term = (vtot - valpha) * dvar - reswt * zeta
            dnllda = np.sum(term, axis=1).T / a
            # Exclude first term, which is held fixed
            gradnll_scaled = np.append(
                gradnll_scaled, 2 * dnllda[1:] * scale_delta_a
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
                gradnll_scaled, 2 * dnlldeta[1:] * scale_eta
            )

    return costfun_scaled / (m * n), gradnll_scaled / (m * n)


def noisefit(
    x: ArrayLike,
    *,
    dt: float | None = None,
    sigma_alpha0: float | None = None,
    sigma_beta0: float | None = None,
    sigma_tau0: float | None = None,
    mu0: ArrayLike | None = None,
    a0: ArrayLike | None = None,
    eta0: ArrayLike | None = None,
    fix_sigma_alpha: bool = False,
    fix_sigma_beta: bool = False,
    fix_sigma_tau: bool = False,
    fix_mu: bool = False,
    fix_a: bool = False,
    fix_eta: bool = False,
    scale_sigma_alpha: float | None = None,
    scale_sigma_beta: float | None = None,
    scale_sigma_tau: float | None = None,
    scale_delta_mu: ArrayLike | None = None,
    scale_delta_a: ArrayLike | None = None,
    scale_eta: ArrayLike | None = None,
) -> NoiseResult:
    r"""
    Estimate noise model from a set of nominally identical waveforms.

    The data array ``x`` should include `m` nominally identical waveform
    measurements, where each waveform comprises `n` samples that are spaced
    equally in time. The ``noise_model`` attribute of the return object
    is an instance of the :class:`NoiseModel` class that represents the
    estimated noise parameters. See `Notes` for details.

    Parameters
    ----------
    x : array_like with shape (n, m)
        Data array composed of ``m`` waveforms, each of which is sampled at
        ``n`` points.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to ``1.0``. In this case, the unit of time is the
        sampling time for the noise model parameter ``sigma_tau``, the delay
        parameter array ``eta``, and the initial guesses for both quantities.
    sigma_alpha0, sigma_beta0, sigma_tau0 : float, optional
        Initial values for noise parameters. When set to
        ``None``, the default, use a linear least-squares fit of the noise
        model to ``np.var(x, 1, ddof=1)``.
    mu0 : array_like with shape(n,), optional
        Initial guess, signal vector with shape (n,). Default is first column
        of ``x``.
    a0 : array_like with shape(m,), optional
        Initial guess, signal amplitude drift vector with shape (m,). Default
        is ``np.ones(m)``.
    eta0 : array_like with shape(m,), optional
        Initial guess, signal delay drift vector with shape (m,). Default is
        ``np.zeros(m)``.
    fix_sigma_alpha, fix_sigma_beta, fix_sigma_tau : bool, optional
        Fix the associated noise parameter. Default is False.
    fix_mu : bool, optional
        Fix signal vector. Default is False.
    fix_a : bool, optional
        Fix signal amplitude drift vector. Default is False.
    fix_eta : bool, optional
        Fix signal delay drift vector. Default is False.

    Returns
    -------
    res : NoiseResult
        Fit result represented as a ``NoiseResult`` object. Important
        attributes are: ``noise_model``, an instance of :class:`NoiseModel`
        with the estimated noise parameters; ``mu``, the estimated signal
        vector; ``a``, the estimated signal amplitude drift vector; ``eta``,
        the estimated signal delay drift vector; ``fval``, the value of the
        optimized NLL cost function; and ``diagnostic``, an instance of
        :class:`scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.minimize`. Note that the fit parameters are
        scaled internally to improve convergence (see `Other Parameters`
        below), which affects the attributes ``fun``, ``jac``, and ``hess_inv``
        of ``diagnostic``. See :class:`NoiseResult` for more details and for a
        description of other attributes.

    Other Parameters
    ----------------
    scale_sigma_alpha, scale_sigma_beta, scale_sigma_tau : float, optional
        Scale for varying noise parameters. Default is ``1.0``.
    scale_delta_mu : array_like with shape(n,), optional
        Scale for varying signal vector. When set to ``None``, the default,
        use ``NoiseModel(sigma_alpha0, sigma_beta0,
        sigma_tau0).amplitude(mu0)``.
    scale_delta_a : array_like with shape(n,), optional
            Scale for varying signal amplitude drift vector. Default is
            ``1.0e-2`` for all entries.
    scale_eta : array_like with shape(n,), optional
            Scale for varying signal delay drift vector. Default is
            ``1.0e-3 / dt`` for all entries.

    Raises
    ------
    ValueError
        If all parameters are held fixed or if the input arrays have
        incompatible shapes.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

    See Also
    --------
    NoiseModel : Noise model class.

    Notes
    -----
    Given an :math:`N\times M` data array :math:`\mathbf{X}`, the function uses
    the BFGS method of :func:`scipy.optimize.minimize` to minimize the
    maximum-likelihood cost function [1]_

    .. math:: \begin{split}\
        Q_\text{ML}\
        (\sigma_\alpha,\sigma_\beta,\sigma_\tau,\boldsymbol{\mu},\
        \mathbf{A},\boldsymbol{\eta};\mathbf{X})\
        = \frac{1}{MN}\sum_{k=0}^{N-1}\sum_{l=0}^{M-1}& \
        \left[\ln\sigma_{kl}^2 + \frac{(X_{kl} \
        - Z_{kl})^2}{\sigma_{kl}^2}\right]\
        \end{split}

    with respect to the unknown model parameters :math:`\sigma_\alpha`,
    :math:`\sigma_\beta`, :math:`\sigma_\tau`, :math:`\boldsymbol{\mu}`,
    :math:`\mathbf{A}` and :math:`\boldsymbol{\eta}`. The model assumes that
    the elements of :math:`\mathbf{X}` are normally distributed around the
    corresponding elements of :math:`\mathbf{Z}` with variances given by
    :math:`\boldsymbol{\sigma}^2`, which are in turn given by

    .. math:: Z_{kl} = A_l\mu(t_k - \eta_l)

    and

    .. math:: \sigma_{kl}^2 = \sigma_\alpha^2 + \sigma_\beta^2 Z_{kl}^2 \
        + \sigma_\tau^2(\mathbf{D}\mathbf{Z})_{kl}^2,

    where :math:`\mathbf{D}` is the time-domain derivative operator. The
    function :math:`\mu(t)` represents an ideal primary waveform, which drifts
    in amplitude and temporal location between each of the :math:`M` waveform
    measurements. Each waveform comprises :math:`N` samples at the nominal
    times :math:`t_k`, and the drift in the amplitude and the temporal location
    of the :math:`l` waveform are given by :math:`A_l` and :math:`\eta_l`,
    respectively, with initial values fixed at :math:`A_0 = 1.0` and
    :math:`\eta_0 = 0.0` by definition.

    References
    ----------
    .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
        Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
        likelihood parameter estimation in terahertz time-domain
        spectroscopy," Opt. Express **29**, 4912-4926 (2021),
        `<https://doi.org/10.1364/OE.417724>`_.

    Examples
    --------
    In basic usage, it should be possible to call ``noisefit`` with just
    the data array. In the code below, we simulate ``m = 50`` noisy waveforms
    with ``n = 256`` time points each, then verify that the fit results are
    consistent with the true noise parameters.

    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> rng = np.random.default_rng(0)
    >>> n, m, dt = 256, 50, 0.05
    >>> thz.set_option("sampling_time", dt)
    >>> t = thz.timebase(n)
    >>> mu = thz.wave(n)
    >>> alpha, beta, tau = 1e-5, 1e-2, 1e-3
    >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
    ...  sigma_tau=tau)
    >>> a = 1.0 + 1e-2 * np.concatenate(([0.0],
    ...                                 rng.standard_normal(m - 1)))
    >>> eta = 1e-3 * np.concatenate(([0.0], rng.standard_normal(m - 1)))
    >>> z = thz.scaleshift(np.repeat(np.atleast_2d(mu), m, axis=0),
    ...                    a=a, eta=eta).T  # Orient the array columnwise
    >>> x = z + noise_model.noise_sim(z, axis=1, seed=1234)
    >>> noise_res = thz.noisefit(x)
    >>> noise_res.noise_model  #doctest: +NORMALIZE_WHITESPACE
    NoiseModel(sigma_alpha=1.000...e-05, sigma_beta=0.009609...,
    sigma_tau=0.0009243..., dt=0.05)

    >>> plt.plot(t, np.std(thz.scaleshift(x, a=1 / noise_res.a,
    ... eta=-noise_res.eta, axis=0), axis=1), "-",
    ... label="Data")
    >>> plt.plot(t, noise_res.noise_model.noise_amp(noise_res.mu)
    ...  * np.sqrt(m / (m + 1)), "--", label="Fit")
    >>> plt.legend()
    >>> plt.xlabel("t (ps)")
    >>> plt.ylabel(r"$\sigma(t)$")
    >>> plt.show()
    """
    x = np.asarray(x, dtype=np.float64)
    dt = _assign_sampling_time(dt)

    parsed = _parse_noisefit_input(
        x,
        dt=dt,
        sigma_alpha0=sigma_alpha0,
        sigma_beta0=sigma_beta0,
        sigma_tau0=sigma_tau0,
        mu0=mu0,
        a0=a0,
        eta0=eta0,
        fix_sigma_alpha=fix_sigma_alpha,
        fix_sigma_beta=fix_sigma_beta,
        fix_sigma_tau=fix_sigma_tau,
        fix_mu=fix_mu,
        fix_a=fix_a,
        fix_eta=fix_eta,
        scale_sigma_alpha=scale_sigma_alpha,
        scale_sigma_beta=scale_sigma_beta,
        scale_sigma_tau=scale_sigma_tau,
        scale_delta_mu=scale_delta_mu,
        scale_delta_a=scale_delta_a,
        scale_eta=scale_eta,
    )

    objective, x0, input_parsed = parsed

    # Minimize cost function with respect to free parameters
    out = minimize(
        objective,
        x0,
        method="BFGS",
        jac=True,
    )
    fit_result = _parse_noisefit_output(out, x, dt=dt, **input_parsed)
    return fit_result


def _parse_noisefit_input(
    x: NDArray[np.float64],
    dt: float,
    *,
    sigma_alpha0: float | None,
    sigma_beta0: float | None,
    sigma_tau0: float | None,
    mu0: ArrayLike | None,
    a0: ArrayLike | None,
    eta0: ArrayLike | None,
    fix_sigma_alpha: bool,
    fix_sigma_beta: bool,
    fix_sigma_tau: bool,
    fix_mu: bool,
    fix_a: bool,
    fix_eta: bool,
    scale_sigma_alpha: float | None,
    scale_sigma_beta: float | None,
    scale_sigma_tau: float | None,
    scale_delta_mu: ArrayLike | None,
    scale_delta_a: ArrayLike | None,
    scale_eta: ArrayLike | None,
) -> tuple[Callable, NDArray[np.float64], dict]:
    """Parse noisefit inputs"""
    if x.ndim != NUM_NOISE_DATA_DIMENSIONS:
        msg = "Data array x must be 2D"
        raise ValueError(msg)
    n, m = x.shape

    if (
        fix_sigma_alpha
        and fix_sigma_beta
        and fix_sigma_tau
        and fix_mu
        and fix_a
        and fix_eta
    ):
        msg = "All variables are fixed"
        raise ValueError(msg)

    # Use the first column of x as a default
    if mu0 is None:
        mu0 = x[:, 0]
    else:
        mu0 = np.asarray(mu0, dtype=np.float64)
        if mu0.size != n:
            msg = "Size of mu0 is incompatible with data array x."
            raise ValueError(msg)

    # Compute time-dependent variance of signal array and
    # the minimum value of the time-dependent noise amplitude
    v_t = np.var(x, 1, ddof=1)
    sigma_min = np.sqrt(np.min(v_t))

    # If any initial guess for the noise parameters is unspecified,
    # estimate all noise parameters with a linear least-squares fit
    # to the time-dependent variance
    if None in [sigma_alpha0, sigma_beta0, sigma_tau0]:
        mu0_f = np.fft.rfft(mu0)
        w = 2 * np.pi * np.fft.rfftfreq(n, dt)
        dmu0_dt = np.fft.irfft(1j * w * mu0_f, n=n)
        a_matrix = np.stack([np.ones(n), mu0**2, dmu0_dt**2], axis=1)
        sol = np.linalg.lstsq(a_matrix, v_t, rcond=None)
        sigma_est = np.ma.sqrt(sol[0]).filled(sigma_min)

    if sigma_alpha0 is None:
        sigma_alpha0 = sigma_est[0]

    if sigma_beta0 is None:
        sigma_beta0 = sigma_est[1]

    if sigma_tau0 is None:
        sigma_tau0 = sigma_est[2]

    noise_model = NoiseModel(sigma_alpha0, sigma_beta0, sigma_tau0, dt=dt)

    if scale_sigma_alpha is None:
        if np.isclose(sigma_alpha0, 0.0):
            scale_sigma_alpha = 1.0
        else:
            scale_sigma_alpha = sigma_alpha0

    if scale_sigma_beta is None:
        if np.isclose(sigma_beta0, 0.0):
            scale_sigma_beta = 1.0
        else:
            scale_sigma_beta = sigma_beta0

    if scale_sigma_tau is None:
        if np.isclose(sigma_tau0, 0.0):
            scale_sigma_tau = 1.0
        else:
            scale_sigma_tau = sigma_tau0 / dt

    if scale_delta_mu is None:
        scale_delta_mu = noise_model.noise_amp(mu0)
        scale_delta_mu[np.isclose(scale_delta_mu, 0.0)] = np.sqrt(
            np.finfo(float).eps
        )

    if scale_delta_a is None:
        scale_delta_a_amp = np.max((sigma_min, sigma_beta0))
        scale_delta_a = scale_delta_a_amp * np.ones(m - 1)

    scale_delta_a = np.asarray(scale_delta_a, dtype=np.float64)

    if scale_eta is None:
        scale_eta_amp = np.max((sigma_min, sigma_tau0))
        scale_eta = scale_eta_amp * np.ones(m - 1) / dt

    scale_eta = np.asarray(scale_eta, dtype=np.float64)

    scale_sigma = np.array(
        [scale_sigma_alpha, scale_sigma_beta, scale_sigma_tau],
        dtype=np.float64,
    )

    # Replace log(x) with -1e2 when x <= 0
    v0_scaled = (
        np.asarray(
            [sigma_alpha0, sigma_beta0, sigma_tau0 / dt], dtype=np.float64
        )
        / scale_sigma
    ) ** 2
    logv0_scaled = np.ma.log(v0_scaled).filled(-1.0e2)
    delta0 = (x[:, 0] - mu0) / scale_delta_mu

    if a0 is None:
        a0 = np.ones(m)
    else:
        a0 = np.asarray(a0, dtype=np.float64)
        if a0.size != m:
            msg = "Size of a0 is incompatible with data array x."
            raise ValueError(msg)

    epsilon0 = (a0[1:] - 1.0) / (a0[0] * scale_delta_a)

    if eta0 is None:
        eta0 = np.zeros(m)
    else:
        eta0 = np.asarray(eta0, dtype=np.float64)
        if eta0.size != m:
            msg = "Size of eta0 is incompatible with data array x."
            raise ValueError(msg)

    eta_on_dt0 = eta0[1:] / dt / scale_eta

    # Set initial guesses for all free parameters
    x0 = np.array([], dtype=np.float64)
    if not fix_sigma_alpha:
        x0 = np.append(x0, logv0_scaled[0])
    if not fix_sigma_beta:
        x0 = np.append(x0, logv0_scaled[1])
    if not fix_sigma_tau:
        x0 = np.append(x0, logv0_scaled[2])
    if not fix_mu:
        x0 = np.concatenate((x0, delta0))
    if not fix_a:
        x0 = np.concatenate((x0, epsilon0))
    if not fix_eta:
        x0 = np.concatenate((x0, eta_on_dt0))

    # Bundle free parameters together into objective function
    def objective(_p):
        if fix_sigma_alpha:
            _logv_alpha = logv0_scaled[0]
        else:
            _logv_alpha = _p[0]
            _p = _p[1:]
        if fix_sigma_beta:
            _logv_beta = logv0_scaled[1]
        else:
            _logv_beta = _p[0]
            _p = _p[1:]
        if fix_sigma_tau:
            _logv_tau = logv0_scaled[2]
        else:
            _logv_tau = _p[0]
            _p = _p[1:]
        if fix_mu:
            _delta = delta0
        else:
            _delta = _p[:n]
            _p = _p[n:]
        if fix_a:
            _epsilon = epsilon0
        else:
            _epsilon = _p[: m - 1]
            _p = _p[m - 1 :]
        if fix_eta:
            _eta_on_dt = eta_on_dt0
        else:
            _eta_on_dt = _p[: m - 1]
        return _costfun_noisefit(
            x.T,
            _logv_alpha,
            _logv_beta,
            _logv_tau,
            _delta,
            _epsilon,
            _eta_on_dt,
            fix_logv_alpha=fix_sigma_alpha,
            fix_logv_beta=fix_sigma_beta,
            fix_logv_tau=fix_sigma_tau,
            fix_delta_mu=fix_mu,
            fix_delta_a=fix_a,
            fix_eta=fix_eta,
            scale_sigma_alpha=scale_sigma_alpha,
            scale_sigma_beta=scale_sigma_beta,
            scale_sigma_tau=scale_sigma_tau,
            scale_delta_mu=scale_delta_mu,
            scale_delta_a=scale_delta_a,
            scale_eta=scale_eta,
        )

    input_parsed = {
        "sigma_alpha0": sigma_alpha0,
        "sigma_beta0": sigma_beta0,
        "sigma_tau0": sigma_tau0,
        "mu0": mu0,
        "a0": a0,
        "eta0": eta0,
        "fix_sigma_alpha": fix_sigma_alpha,
        "fix_sigma_beta": fix_sigma_beta,
        "fix_sigma_tau": fix_sigma_tau,
        "fix_mu": fix_mu,
        "fix_a": fix_a,
        "fix_eta": fix_eta,
        "scale_sigma_alpha": scale_sigma_alpha,
        "scale_sigma_beta": scale_sigma_beta,
        "scale_sigma_tau": scale_sigma_tau,
        "scale_delta_mu": scale_delta_mu,
        "scale_delta_a": scale_delta_a,
        "scale_eta": scale_eta,
    }
    return objective, x0, input_parsed


def _parse_noisefit_output(
    out: OptimizeResult,
    x: NDArray[np.float64],
    dt: float,
    *,
    sigma_alpha0: float,
    sigma_beta0: float,
    sigma_tau0: float,
    mu0: NDArray[np.float64],
    a0: NDArray[np.float64],
    eta0: NDArray[np.float64],
    fix_sigma_alpha: bool,
    fix_sigma_beta: bool,
    fix_sigma_tau: bool,
    fix_mu: bool,
    fix_a: bool,
    fix_eta: bool,
    scale_sigma_alpha: float,
    scale_sigma_beta: float,
    scale_sigma_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta: NDArray[np.float64],
) -> NoiseResult:
    """Parse noisefit output"""
    # Parse output
    n, m = x.shape

    x_out = out.x
    if fix_sigma_alpha:
        alpha = sigma_alpha0
    else:
        alpha = np.exp(x_out[0] / 2) * scale_sigma_alpha
        x_out = x_out[1:]
    if fix_sigma_beta:
        beta = sigma_beta0
    else:
        beta = np.exp(x_out[0] / 2) * scale_sigma_beta
        x_out = x_out[1:]
    if fix_sigma_tau:
        tau = sigma_tau0
    else:
        tau = np.exp(x_out[0] / 2) * scale_sigma_tau * dt
        x_out = x_out[1:]
    # noinspection PyArgumentList
    noise_model = NoiseModel(alpha, beta, tau, dt=dt)

    if fix_mu:
        mu_out = mu0
    else:
        mu_out = x[:, 0] - x_out[:n] * scale_delta_mu
        x_out = x_out[n:]

    if fix_a:
        a_out = a0
    else:
        a_out = np.concatenate(([1.0], 1.0 + x_out[: m - 1] * scale_delta_a))
        x_out = x_out[m - 1 :]

    if fix_eta:
        eta_out = eta0
    else:
        eta_out = np.concatenate(([0.0], x_out[: m - 1] * dt * scale_eta))

    diagnostic = out
    fun = out.fun

    # Concatenate scaling vectors for all sets of free parameters
    scale_hess_inv = np.concatenate(
        [
            val
            for tf, val in zip(
                [
                    fix_sigma_alpha,
                    fix_sigma_beta,
                    fix_sigma_tau,
                    fix_mu,
                    fix_a,
                    fix_eta,
                ],
                [
                    [scale_sigma_alpha],
                    [scale_sigma_beta],
                    [scale_sigma_tau],
                    scale_delta_mu,
                    scale_delta_a,
                    scale_eta,
                ],
            )
            if not tf
        ]
    )

    # Convert inverse Hessian into unscaled parameters
    hess_inv = np.diag(scale_hess_inv) @ out.hess_inv @ np.diag(scale_hess_inv)

    # Determine parameter uncertainty vector from diagonal entries
    err = np.sqrt(np.diag(hess_inv))
    err_mu = np.array([])
    err_a = np.array([])
    err_eta = np.array([])

    # Parse error vector
    # Propagate error from log(V) to sigma
    if not fix_sigma_alpha:
        err_sigma_alpha = np.sqrt(0.5 * alpha * err[0])
        err = err[1:]
    else:
        err_sigma_alpha = 0.0

    if not fix_sigma_beta:
        err_sigma_beta = np.sqrt(0.5 * beta * err[0])
        err = err[1:]
    else:
        err_sigma_beta = 0.0

    if not fix_sigma_tau:
        err_sigma_tau = np.sqrt(0.5 * tau * err[0])
        err = err[1:]
    else:
        err_sigma_tau = 0.0

    if not fix_mu:
        err_mu = err[:n]
        err = err[n:]

    if not fix_a:
        err_a = np.concatenate(([0], err[: m - 1]))
        err = err[m - 1 :]

    if not fix_eta:
        err_eta = np.concatenate(([0], err[: m - 1]))

    return NoiseResult(
        noise_model,
        mu_out,
        a_out,
        eta_out,
        fun,
        hess_inv,
        err_sigma_alpha,
        err_sigma_beta,
        err_sigma_tau,
        err_mu,
        err_a,
        err_eta,
        diagnostic,
    )


def fit(
    fun: Callable,
    p0: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    *,
    dt: float | None = None,
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

            ``fun(p, w, *args, **kwargs) -> ndarray``

        Assumes the :math:`+i\omega t` convention for harmonic time dependence.
    p0 : array_like
        Initial guess for ``p``.
    x : array_like
        Measured input signal.
    y : array_like
        Measured output signal.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is None, which sets
        the sampling time to ``thztools.options.sampling_time``. If both
        ``dt`` and ``thztools.options.sampling_time`` are ``None``, the
        sampling time is set to ``1.0``. In this case, the angular frequency
        ``omega`` must be given in units of radians per sampling time, and any
        parameters in ``args`` must be expressed with the sampling time as the
        unit of time.
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

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.
    """
    fit_method = "trf"

    p0 = np.asarray(p0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sigma_parms = np.asarray(sigma_parms, dtype=np.float64)

    dt = _assign_sampling_time(dt)

    n = y.shape[-1]
    n_p = len(p0)

    if p_bounds is None:
        p_bounds = (-np.inf, np.inf)
        fit_method = "lm"
    else:
        p_bounds = np.asarray(p_bounds, dtype=np.float64)
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
    w_bounds = 2 * np.pi * np.asarray(f_bounds, dtype=np.float64)
    w_below_idx = w <= w_bounds[0]
    w_above_idx = w > w_bounds[1]
    w_in_idx = np.invert(w_below_idx) * np.invert(w_above_idx)
    n_f = len(w)

    alpha, beta, tau = sigma_parms
    # noinspection PyArgumentList
    noise_model = NoiseModel(alpha, beta, tau, dt=dt)
    sigma_x = noise_model.noise_amp(x)
    sigma_y = noise_model.noise_amp(y)
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
