from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from numpy import pi
from numpy.fft import irfft, rfft, rfftfreq
from numpy.random import default_rng
from scipy import signal
from scipy.linalg import sqrtm
from scipy.optimize import OptimizeResult, approx_fprime, minimize

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

NUM_NOISE_PARAMETERS = 3
NUM_NOISE_DATA_DIMENSIONS = 2


@dataclass
class GlobalOptions:
    r"""
    Class for storing global options.

    Attributes
    ----------
    sampling_time : float | None, optional
        Global sampling time, normally in picoseconds. When set to None, the
        default, times and frequencies are treated as dimensionless quantities
        that are scaled by the (undetermined) sampling time.
    """

    sampling_time: float | None = None


#: Instance of ``GlobalOptions`` that stores global options.
options = GlobalOptions()


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

    Notes
    -----
    Available options, with descriptions:

    sampling_time : float | None
        Global sampling time, normally in picoseconds. When set to ``None``,
        the default, times and frequencies are treated as dimensionless
        quantities that are scaled by the (undetermined) sampling time.

    Examples
    --------
    The global ``sampling_time`` option is initialized to ``None`` at startup.

    >>> import thztools as thz
    >>> thz.get_option("sampling_time")

    Use the :func:`set_option` function to set the global sampling time to the
    preferred value.

    >>> thz.set_option("sampling_time", 0.05)
    >>> thz.get_option("sampling_time")
    0.05
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

    Notes
    -----
    Available options, with descriptions:

    sampling_time : float | None
        Global sampling time, normally in picoseconds. When set to ``None``,
        the default, times and frequencies are treated as dimensionless
        quantities that are scaled by the (undetermined) sampling time.

    Examples
    --------
    The global ``sampling_time`` option is initialized to ``None`` at startup.

    >>> import thztools as thz
    >>> thz.get_option("sampling_time")

    Set the global sampling time to a preferred value.

    >>> thz.set_option("sampling_time", 0.05)
    >>> thz.get_option("sampling_time")
    0.05
    """
    return getattr(options, key)


def reset_option(key: str | None = None) -> None:
    r"""
    Reset one or more global options to default values.

    Parameters
    ----------
    key : str, optional
        Option name. When ``None``, the default, resets all options.

    Returns
    -------
    None
        No return value.

    Notes
    -----
    Available options, with descriptions:

    sampling_time : float | None
        Global sampling time, normally in picoseconds. When set to ``None``,
        the default, times and frequencies are treated as dimensionless
        quantities that are scaled by the (undetermined) sampling time.

    Examples
    --------
    The global ``sampling_time`` option is initialized to ``None`` at startup.

    >>> import thztools as thz
    >>> thz.get_option("sampling_time")

    Use the :func:`set_option` function to change the value from the default.

    >>> thz.set_option("sampling_time", 0.05)
    >>> thz.get_option("sampling_time")
    0.05

    Reset all options to default values.

    >>> thz.reset_option()
    >>> thz.get_option("sampling_time")
    """
    default_options = GlobalOptions()
    if key is not None:
        val = getattr(default_options, key)
        set_option(key, val)
    else:
        for local_key in GlobalOptions.__dataclass_fields__:
            local_val = getattr(default_options, local_key)
            set_option(local_key, local_val)


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
    >>> t = thz.timebase(n, dt=dt)
    >>> mu = thz.wave(n)
    >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
    >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
    ...  sigma_tau=tau, dt=dt)
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
            Time-domain signal array.
        axis : int, optional
            Signal axis. Default is the last axis in ``x``.

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
        >>> t = thz.timebase(n, dt=dt)
        >>> mu = thz.wave(n, dt=dt)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau, dt=dt)
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
        if x.ndim > 1 and axis != -1:
            x = np.moveaxis(x, axis, -1)

        n = x.shape[-1]
        w_scaled = 2 * pi * rfftfreq(n)
        xdot = irfft(1j * w_scaled * rfft(x), n=n) / dt

        noise_variance = (
            self.sigma_alpha**2
            + (self.sigma_beta * x) ** 2
            + (self.sigma_tau * xdot) ** 2
        )

        if x.ndim > 1 and axis != -1:
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
            Time-domain signal array.
        axis : int, optional
            Signal axis. Default is the last axis in ``x``.

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
        >>> t = thz.timebase(n, dt=dt)
        >>> mu = thz.wave(n, dt=dt)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau, dt=dt)
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
            Time-domain signal array.
        axis : int, optional
            Signal axis. Default is the last axis in ``x``.
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
        >>> t = thz.timebase(n, dt=dt)
        >>> mu = thz.wave(n, dt=dt)
        >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
        >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
        ... sigma_tau=tau, dt=dt)
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
        if x.ndim > 1 and axis != -1:
            x = np.moveaxis(x, axis, -1)

        amp = self.noise_amp(x)
        rng = default_rng(seed)
        noise = amp * rng.standard_normal(size=x.shape)
        if x.ndim > 1 and axis != -1:
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
def apply_frf(
    frfun: Callable,
    x: ArrayLike,
    *,
    dt: float | None = None,
    numpy_sign_convention: bool = True,
    args: ArrayLike = (),
) -> NDArray[np.float64]:
    r"""
    Apply a frequency response function to a waveform.

    Parameters
    ----------
    frfun : callable
        Frequency response function.

            ``frfun(omega, *args) -> ndarray``

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
        Adopt NumPy sign convention for harmonic time dependence, e.g., express
        a harmonic function with frequency :math:`\omega` as
        :math:`x(t) = a e^{i\omega t}`. Default is ``True``. When set to
        ``False``, uses the convention more common in physics,
        :math:`x(t) = a e^{-i\omega t}`.
    args : array_like, optional
        Extra arguments passed to the frequency response function. All elements
        must be real quantities.

    Returns
    -------
    y : ndarray
        Result of applying the frequency response function to ``x``.

    Warns
    -----
    UserWarning
        If ``thztools.options.sampling_time`` and the ``dt`` parameter
        are both not ``None`` and are set to different ``float`` values, the
        function will set the sampling time to ``dt`` and raise a
        :class:`UserWarning`.

    See Also
    --------
    fit : Fit a frequency response function to time-domain data.

    Notes
    -----
    The output waveform is computed by transforming :math:`x[n]` into the
    frequency domain, multiplying by the frequency response function
    :math:`H[n]`, then transforming back into the time domain.

    .. math:: y[n] = \mathcal{F}^{-1}\{H[n] \mathcal{F}\{x[n]\}\}

    Examples
    --------
    Apply a frequency response function that rescales the input by :math:`a`
    and shifts it by :math:`\tau`.

    .. math:: H(\omega) = a\exp(-i\omega\tau).

    Note that this form assumes the :math:`e^{+i\omega t}` representation of
    harmonic time dependence, which corresponds to the default setting
    ``numpy_sign_convention=True``.

    >>> import numpy as np
    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> t = thz.timebase(n, dt=dt)
    >>> x = thz.wave(n, dt=dt)
    >>> def shiftscale(_w, _a, _tau):
    ...     return _a * np.exp(-1j * _w * _tau)
    >>>
    >>> y = thz.apply_frf(
    ...     shiftscale, x, dt=dt, numpy_sign_convention=True, args=(0.5, 1)
    ... )

    >>> _, ax = plt.subplots()
    >>>
    >>> ax.plot(t, x, label="x")
    >>> ax.plot(t, y, label="y")
    >>> ax.legend()
    >>> ax.set_xlabel("t (ps)")
    >>> ax.set_ylabel("Amplitude (arb. units)")
    >>> plt.show()

    If the frequency response function is expressed using the
    :math:`e^{-i\omega t}` representation more common in physics,

    .. math:: H(\omega) = a\exp(i\omega\tau),

    set ``numpy_sign_convention=False``.

    >>> def shiftscale_phys(_w, _a, _tau):
    ...     return _a * np.exp(1j * _w * _tau)
    >>>
    >>> y_p = thz.apply_frf(
    ...     shiftscale_phys, x, dt=dt, numpy_sign_convention=False, args=(0.5, 1)
    ... )

    >>> _, ax = plt.subplots()
    >>>
    >>> ax.plot(t, x, label="x")
    >>> ax.plot(t, y_p, label="y")
    >>>
    >>> ax.legend()
    >>> ax.set_xlabel("t (ps)")
    >>> ax.set_ylabel("Amplitude (arb. units)")
    >>>
    >>> plt.show()
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        msg = "x must be a one-dimensional array"
        raise ValueError(msg)

    args = np.atleast_1d(args)

    dt = _assign_sampling_time(dt)
    n = x.size
    w_scaled = 2 * pi * rfftfreq(n)
    h = frfun(w_scaled / dt, *args)
    if numpy_sign_convention:
        y = np.fft.irfft(np.fft.rfft(x) * h, n=n)
    else:
        y = np.fft.irfft(np.fft.rfft(x) * np.conj(h), n=n)

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
    >>> t = thz.timebase(n, dt=dt)
    >>> mu = thz.wave(n, dt=dt)

    >>> _, ax = plt.subplots(layout="constrained")
    >>> ax.plot(t, mu)
    >>> ax.set_xlabel("t (ps)")
    >>> ax.set_ylabel(r"$\mu/\mu_0$")
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

    w = 2 * pi * f_scaled / dt
    ell = np.exp(-((w * taul) ** 2) / 2) / np.sqrt(2 * pi * taul**2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (ell * r) ** 2 * np.exp(1j * w * t0)

    x_unscaled = irfft(np.conj(s), n=n)

    return a * x_unscaled / np.max(x_unscaled)


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

    >>> import numpy as np
    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> t = thz.timebase(n, dt=dt)
    >>> mu = thz.wave(n, dt=dt)
    >>> m = 4
    >>> x = np.repeat(np.atleast_2d(mu), m, axis=0)
    >>> a = 0.5 ** np.arange(m)
    >>> eta = np.arange(m)
    >>> x_adj = thz.scaleshift(x, a=a, eta=eta, dt=dt)

    >>> plt.plot(t, x_adj.T, label=[f"{k=}" for k in range(4)])
    >>> plt.legend()
    >>> plt.xlabel("t (ps)")
    >>> plt.ylabel(r"$x_{\mathrm{adj}, k}$")
    >>> plt.show()
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.empty(x.shape)

    dt = _assign_sampling_time(dt)

    axis = int(axis)
    if x.ndim > 1 and axis != -1:
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
    w = 2 * pi * f_scaled / dt
    phase = np.expand_dims(eta, axis=eta.ndim) * w

    x_adjusted = np.fft.irfft(
        np.fft.rfft(x) * np.exp(-1j * phase), n=n
    ) * np.expand_dims(a, axis=a.ndim)

    if x.ndim > 1 and axis != -1:
        x_adjusted = np.moveaxis(x_adjusted, -1, axis)

    return x_adjusted


@dataclass
class CommonNLL:
    ressq: NDArray[np.float64]
    vtot: NDArray[np.float64]
    zeta: NDArray[np.float64]
    dzeta: NDArray[np.float64]
    a: NDArray[np.float64]
    zeta_f: NDArray[np.complex128]
    exp_iweta: NDArray[np.complex128]


def _nll_common(
    x: NDArray[np.float64],
    logv_alpha_scaled: float,
    logv_beta_scaled: float,
    logv_tau_scaled: float,
    delta_mu_scaled: NDArray[np.float64],
    delta_a_scaled: NDArray[np.float64],
    eta_on_dt_scaled: NDArray[np.float64],
    *,
    scale_logv_alpha: float,
    scale_logv_beta: float,
    scale_logv_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta_on_dt: NDArray[np.float64],
) -> CommonNLL:
    _, n = x.shape

    valpha = np.exp(logv_alpha_scaled * scale_logv_alpha)
    vbeta = np.exp(logv_beta_scaled * scale_logv_beta)
    vtau = np.exp(logv_tau_scaled * scale_logv_tau)

    mu = x[0, :] - delta_mu_scaled * scale_delta_mu
    a = 1.0 + np.insert(delta_a_scaled * scale_delta_a, 0, 0.0)
    eta_on_dt = np.insert(eta_on_dt_scaled * scale_eta_on_dt, 0, 0.0)

    # Compute frequency vector and Fourier coefficients of mu
    f = rfftfreq(n)
    w = 2 * pi * f
    mu_f = rfft(mu)

    exp_iweta = np.exp(1j * np.outer(eta_on_dt, w))
    zeta_f = ((np.conj(exp_iweta) * mu_f).T * a).T

    zeta = irfft(zeta_f, n=n)
    dzeta = irfft(1j * w * zeta_f, n=n)

    res = x - zeta
    ressq = res**2
    vtot = valpha + vbeta * zeta**2 + vtau * dzeta**2

    return CommonNLL(
        ressq=ressq,
        vtot=vtot,
        zeta=zeta,
        dzeta=dzeta,
        a=a,
        zeta_f=zeta_f,
        exp_iweta=exp_iweta,
    )


def _nll_noisefit(
    x: NDArray[np.float64],
    logv_alpha_scaled: float,
    logv_beta_scaled: float,
    logv_tau_scaled: float,
    delta_mu_scaled: NDArray[np.float64],
    delta_a_scaled: NDArray[np.float64],
    eta_on_dt_scaled: NDArray[np.float64],
    *,
    scale_logv_alpha: float,
    scale_logv_beta: float,
    scale_logv_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta_on_dt: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    r"""
    Compute the cost function for the time-domain noise model.

    Computes the maximum-likelihood cost function for obtaining the
    data matrix ``x`` given ``logv_alpha_scaled``, ``logv_beta_scaled``,
    ``logv_tau_scaled``, ``delta_mu_scaled``, ``delta_a_scaled``,
    ``eta_on_dt_scaled``, ``scale_logv_alpha``, ``scale_logv_beta``,
    ``scale_logv_tau``, ``scale_delta_mu``, ``scale_delta_a``, and
    ``scale_eta_on_dt``.

    Parameters
    ----------
    x : ndarray
        Data matrix with shape (m, n), row-oriented.
    logv_alpha_scaled, logv_beta_scaled, logv_tau_scaled : float
        Logarithm of the associated scaled noise variance parameter.
    delta_mu_scaled : ndarray
        Scaled signal deviation vector with shape (n,).
    delta_a_scaled: ndarray
        Scaled amplitude deviation vector with shape (m - 1,).
    eta_on_dt_scaled : ndarray
        Scaled delay deviation vector with shape (m - 1,).
    scale_logv_alpha, scale_logv_beta,  scale_logv_tau: float
        Scale parameters for log variance parameters.
    scale_delta_mu : ndarray
        Array of scale parameters for ``delta`` with shape (n,).
    scale_delta_a : ndarray
        Array of scale parameters for ``alpha`` with shape (m - 1,).
    scale_eta_on_dt : ndarray
        Array of scale parameters for ``eta`` with shape (m - 1,). Should be
        expressed in terms of the sampling time, i.e.,
        ``scale_sigma_tau_on_dt = scale_sigma_tau / dt``, where ``dt`` is the
        sampling time and ``scale_eta`` is the scale factor used in
        ``eta_scaled = eta / scale_eta``.

    Returns
    -------
    nll : float
        Negative log-likelihood, with constant offset :math:`(MN/2)\ln(2\pi)`
        subtracted.
    """
    common = _nll_common(
        x=x,
        logv_alpha_scaled=logv_alpha_scaled,
        logv_beta_scaled=logv_beta_scaled,
        logv_tau_scaled=logv_tau_scaled,
        delta_mu_scaled=delta_mu_scaled,
        delta_a_scaled=delta_a_scaled,
        eta_on_dt_scaled=eta_on_dt_scaled,
        scale_logv_alpha=scale_logv_alpha,
        scale_logv_beta=scale_logv_beta,
        scale_logv_tau=scale_logv_tau,
        scale_delta_mu=scale_delta_mu,
        scale_delta_a=scale_delta_a,
        scale_eta_on_dt=scale_eta_on_dt,
    )
    ressq = common.ressq
    vtot = common.vtot
    resnormsq_scaled = ressq / vtot

    return 0.5 * (np.sum(np.log(vtot)) + np.sum(resnormsq_scaled))


def _jac_noisefit(
    x: NDArray[np.float64],
    logv_alpha_scaled: float,
    logv_beta_scaled: float,
    logv_tau_scaled: float,
    delta_mu_scaled: NDArray[np.float64],
    delta_a_scaled: NDArray[np.float64],
    eta_on_dt_scaled: NDArray[np.float64],
    *,
    fix_logv_alpha: bool,
    fix_logv_beta: bool,
    fix_logv_tau: bool,
    fix_delta_mu: bool,
    fix_delta_a: bool,
    fix_eta: bool,
    scale_logv_alpha: float,
    scale_logv_beta: float,
    scale_logv_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta_on_dt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Compute the Jacobian of ``_nll_noisefit`` w.r.t. the free parameters.

    Parameters
    ----------
    x : ndarray
        Data matrix with shape (m, n), row-oriented.
    logv_alpha_scaled, logv_beta_scaled, logv_tau_scaled : float
        Logarithm of the associated scaled noise variance parameter.
    delta_mu_scaled : ndarray
        Scaled signal deviation vector with shape (n,).
    delta_a_scaled: ndarray
        Scaled amplitude deviation vector with shape (m - 1,).
    eta_on_dt_scaled : ndarray
        Scaled delay deviation vector with shape (m - 1,).
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
    scale_logv_alpha, scale_logv_beta,  scale_logv_tau: float
        Scale parameters for log variance parameters.
    scale_delta_mu : ndarray
        Array of scale parameters for ``delta`` with shape (n,).
    scale_delta_a : ndarray
        Array of scale parameters for ``alpha`` with shape (m - 1,).
    scale_eta_on_dt : ndarray
        Array of scale parameters for ``eta`` with shape (m - 1,). Should be
        expressed in terms of the sampling time, i.e.,
        ``scale_sigma_tau_on_dt = scale_sigma_tau / dt``, where ``dt`` is the
        sampling time and ``scale_eta`` is the scale factor used in
        ``eta_scaled = eta / scale_eta``.

    Returns
    -------
    gradnll_scaled : ndarray
        Gradient of the negative log-likelihood function with respect to
        the free parameters.
    """
    m, n = x.shape

    valpha = np.exp(logv_alpha_scaled * scale_logv_alpha)
    vbeta = np.exp(logv_beta_scaled * scale_logv_beta)
    vtau = np.exp(logv_tau_scaled * scale_logv_tau)

    f = rfftfreq(n)
    w = 2 * pi * f

    common = _nll_common(
        x=x,
        logv_alpha_scaled=logv_alpha_scaled,
        logv_beta_scaled=logv_beta_scaled,
        logv_tau_scaled=logv_tau_scaled,
        delta_mu_scaled=delta_mu_scaled,
        delta_a_scaled=delta_a_scaled,
        eta_on_dt_scaled=eta_on_dt_scaled,
        scale_logv_alpha=scale_logv_alpha,
        scale_logv_beta=scale_logv_beta,
        scale_logv_tau=scale_logv_tau,
        scale_delta_mu=scale_delta_mu,
        scale_delta_a=scale_delta_a,
        scale_eta_on_dt=scale_eta_on_dt,
    )
    ressq = common.ressq
    vtot = common.vtot
    zeta = common.zeta
    dzeta = common.dzeta
    zeta_f = common.zeta_f
    a = common.a
    exp_iweta = common.exp_iweta

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    reswt = res / vtot
    dvar = (vtot - ressq) / vtot**2

    # Construct Jacobian subarrays
    if fix_logv_alpha:
        jac_logv_alpha = []
    else:
        jac_logv_alpha = [0.5 * np.sum(dvar) * valpha * scale_logv_alpha]

    if fix_logv_beta:
        jac_logv_beta = []
    else:
        jac_logv_beta = [
            0.5 * np.sum(zeta**2 * dvar) * vbeta * scale_logv_beta
        ]

    if fix_logv_tau:
        jac_logv_tau = []
    else:
        jac_logv_tau = [0.5 * np.sum(dzeta**2 * dvar) * vbeta * scale_logv_tau]

    if fix_delta_mu:
        jac_delta_mu = []
    else:
        p = rfft(vbeta * dvar * zeta - reswt) - 1j * vtau * w * rfft(
            dvar * dzeta
        )
        jac_delta_mu = (
            -np.sum((irfft(exp_iweta * p, n=n).T * a).T, axis=0)
            * scale_delta_mu
        )

    if fix_delta_a:
        jac_delta_a = []
    else:
        term = (vtot - valpha) * dvar - reswt * zeta
        dnllda = np.sum(term, axis=1) / a
        # Exclude first term, which is held fixed
        jac_delta_a = dnllda[1:] * scale_delta_a

    if fix_eta:
        jac_eta = []
    else:
        ddzeta = irfft(-(w**2) * zeta_f, n=n)
        dnlldeta = -np.sum(
            dvar * (zeta * dzeta * valpha + dzeta * ddzeta * vtau)
            - reswt * dzeta,
            axis=1,
        )
        # Exclude first term, which is held fixed
        jac_eta = dnlldeta[1:] * scale_eta_on_dt

    # Concatenate subarrays to produce full Jacobian wrt free parameters
    return np.concatenate(
        (
            jac_logv_alpha,
            jac_logv_beta,
            jac_logv_tau,
            jac_delta_mu,
            jac_delta_a,
            jac_eta,
        )
    )


def _hess_noisefit(
    x: NDArray[np.float64],
    logv_alpha_scaled: float,
    logv_beta_scaled: float,
    logv_tau_scaled: float,
    delta_mu_scaled: NDArray[np.float64],
    delta_a_scaled: NDArray[np.float64],
    eta_on_dt_scaled: NDArray[np.float64],
    *,
    fix_logv_alpha: bool,
    fix_logv_beta: bool,
    fix_logv_tau: bool,
    fix_delta_mu: bool,
    fix_delta_a: bool,
    fix_eta: bool,
    scale_logv_alpha: float,
    scale_logv_beta: float,
    scale_logv_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta_on_dt: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""
    Compute the Hessian of ``_nll_noisefit`` w.r.t. the free parameters.

    Parameters
    ----------
    x : ndarray
        Data matrix with shape (m, n), row-oriented.
    logv_alpha_scaled, logv_beta_scaled, logv_tau_scaled : float
        Logarithm of the associated scaled noise variance parameter.
    delta_mu_scaled : ndarray
        Scaled signal deviation vector with shape (n,).
    delta_a_scaled: ndarray
        Scaled amplitude deviation vector with shape (m - 1,).
    eta_on_dt_scaled : ndarray
        Scaled delay deviation vector with shape (m - 1,).
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
    scale_logv_alpha, scale_logv_beta,  scale_logv_tau: float
        Scale parameters for log variance parameters.
    scale_delta_mu : ndarray
        Array of scale parameters for ``delta`` with shape (n,).
    scale_delta_a : ndarray
        Array of scale parameters for ``alpha`` with shape (m - 1,).
    scale_eta_on_dt : ndarray
        Array of scale parameters for ``eta`` with shape (m - 1,). Should be
        expressed in terms of the sampling time, i.e.,
        ``scale_sigma_tau_on_dt = scale_sigma_tau / dt``, where ``dt`` is the
        sampling time and ``scale_eta`` is the scale factor used in
        ``eta_scaled = eta / scale_eta``.

    Returns
    -------
    gradnll_scaled : ndarray
        Gradient of the negative log-likelihood function with respect to
        the free parameters.
    """
    m, n = x.shape

    valpha = np.exp(logv_alpha_scaled * scale_logv_alpha)
    vbeta = np.exp(logv_beta_scaled * scale_logv_beta)
    vtau = np.exp(logv_tau_scaled * scale_logv_tau)

    f = rfftfreq(n)
    w = 2 * pi * f

    common = _nll_common(
        x=x,
        logv_alpha_scaled=logv_alpha_scaled,
        logv_beta_scaled=logv_beta_scaled,
        logv_tau_scaled=logv_tau_scaled,
        delta_mu_scaled=delta_mu_scaled,
        delta_a_scaled=delta_a_scaled,
        eta_on_dt_scaled=eta_on_dt_scaled,
        scale_logv_alpha=scale_logv_alpha,
        scale_logv_beta=scale_logv_beta,
        scale_logv_tau=scale_logv_tau,
        scale_delta_mu=scale_delta_mu,
        scale_delta_a=scale_delta_a,
        scale_eta_on_dt=scale_eta_on_dt,
    )
    # Compute residuals and their squares for subsequent computations
    ressq = common.ressq
    vtot = common.vtot
    zeta = common.zeta
    dzeta = common.dzeta
    zeta_f = common.zeta_f
    a = common.a
    exp_iweta = common.exp_iweta

    ddzeta = irfft(-(w**2) * zeta_f, n=n)
    dddzeta = irfft(-1j * (w**3) * zeta_f, n=n)

    res = x - zeta
    dvar = (vtot - ressq) / vtot**2
    ddvar = (2 * ressq - vtot) / vtot**3

    dzeta_dmu = irfft(
        a[:, np.newaxis, np.newaxis]
        * np.conj(exp_iweta)[:, np.newaxis, :]
        * rfft(np.eye(n))[np.newaxis, :, :],
        n=n,
    )

    ddzeta_dmu = irfft(
        a[:, np.newaxis, np.newaxis]
        * 1j
        * w
        * np.conj(exp_iweta)[:, np.newaxis, :]
        * rfft(np.eye(n))[np.newaxis, :, :],
        n=n,
    )

    dddzeta_dmu = irfft(
        a[:, np.newaxis, np.newaxis]
        * -(w**2)
        * np.conj(exp_iweta)[:, np.newaxis, :]
        * rfft(np.eye(n))[np.newaxis, :, :],
        n=n,
    )

    # Hessian block for (logv, logv)
    if fix_logv_alpha:
        h_va_va = np.atleast_2d([])
    else:
        h_va_va = np.atleast_2d(
            [0.5 * valpha * np.sum(dvar) + 0.5 * valpha**2 * np.sum(ddvar)]
        )

    if fix_logv_alpha or fix_logv_beta:
        h_va_vb = np.atleast_2d([])
    else:
        h_va_vb = np.atleast_2d(
            [0.5 * valpha * vbeta * np.sum(ddvar * zeta**2)]
        )

    if fix_logv_alpha or fix_logv_tau:
        h_va_vt = np.atleast_2d([])
    else:
        h_va_vt = np.atleast_2d(
            [0.5 * valpha * vtau * np.sum(ddvar * dzeta**2)]
        )

    if fix_logv_beta:
        h_vb_vb = np.atleast_2d([])
    else:
        h_vb_vb = np.atleast_2d(
            [
                0.5 * vbeta * np.sum(dvar * zeta**2)
                + 0.5 * vbeta**2 * np.sum(ddvar * zeta**4)
            ]
        )

    if fix_logv_beta or fix_logv_tau:
        h_vb_vt = np.atleast_2d([])
    else:
        h_vb_vt = np.atleast_2d(
            [0.5 * vbeta * vtau * np.sum(ddvar * zeta**2 * dzeta**2)]
        )

    if fix_logv_tau:
        h_vt_vt = np.atleast_2d([])
    else:
        h_vt_vt = np.atleast_2d(
            [
                0.5 * vtau * np.sum(dvar * dzeta**2)
                + 0.5 * vtau**2 * np.sum(ddvar * dzeta**4)
            ]
        )

    # Hessian block for (logv, delta_mu)
    if fix_logv_alpha or fix_delta_mu:
        h_va_mu = np.atleast_2d([])
    else:
        h_va_mu = np.atleast_2d(
            np.einsum(
                "jk, jpk",
                ddvar * valpha * vbeta * zeta + res * valpha / vtot**2,
                dzeta_dmu,
            )
            + np.einsum("jk, jpk", ddvar * valpha * vtau * dzeta, ddzeta_dmu)
        )

    if fix_logv_beta or fix_delta_mu:
        h_vb_mu = np.atleast_2d([])
    else:
        h_vb_mu = np.atleast_2d(
            np.einsum(
                "jk, jpk",
                dvar * vbeta * zeta
                + ddvar * vbeta**2 * zeta**3
                + res * vbeta * zeta**2 / vtot**2,
                dzeta_dmu,
            )
            + np.einsum(
                "jk, jpk", ddvar * vbeta * vtau * zeta**2 * dzeta, ddzeta_dmu
            )
        )

    if fix_logv_tau or fix_delta_mu:
        h_vt_mu = np.atleast_2d([])
    else:
        h_vt_mu = np.atleast_2d(
            np.einsum(
                "jk, jpk",
                ddvar * vbeta * vtau * zeta * dzeta**2
                + res * vtau * dzeta**2 / vtot**2,
                dzeta_dmu,
            )
            + np.einsum(
                "jk, jpk",
                dvar * vtau * dzeta + ddvar * vtau**2 * dzeta**3,
                ddzeta_dmu,
            )
        )

    # Hessian block for (log_v, delta_a)
    if fix_logv_alpha or fix_delta_a:
        h_va_a = np.atleast_2d([])
    else:
        h_va_a = np.atleast_2d(
            np.sum(
                (
                    ddvar * valpha * (vbeta * zeta**2 + vtau * dzeta**2)
                    + res * valpha * zeta / vtot**2
                )[1:, :],
                axis=1,
            )
            / a[1:]
        )

    if fix_logv_beta or fix_delta_a:
        h_vb_a = np.atleast_2d([])
    else:
        h_vb_a = np.atleast_2d(
            np.sum(
                (
                    dvar * vbeta * zeta**2
                    + ddvar
                    * vbeta
                    * zeta**2
                    * (vbeta * zeta**2 + vtau * dzeta**2)
                    + res * vbeta * zeta**3 / vtot**2
                )[1:, :],
                axis=1,
            )
            / a[1:]
        )

    if fix_logv_tau or fix_delta_a:
        h_vt_a = np.atleast_2d([])
    else:
        h_vt_a = np.atleast_2d(
            np.sum(
                (
                    dvar * vtau * dzeta**2
                    + ddvar
                    * vtau
                    * dzeta**2
                    * (vbeta * zeta**2 + vtau * dzeta**2)
                    + res * vtau * zeta * dzeta**2 / vtot**2
                )[1:, :],
                axis=1,
            )
            / a[1:]
        )

    # Hessian block for (log_v, eta)
    if fix_logv_alpha or fix_eta:
        h_va_eta = np.atleast_2d([])
    else:
        h_va_eta = -np.atleast_2d(
            np.sum(
                ddvar * valpha * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                + res * valpha * dzeta / vtot**2,
                axis=1,
            )[1:]
        )

    if fix_logv_beta or fix_eta:
        h_vb_eta = np.atleast_2d([])
    else:
        h_vb_eta = -np.atleast_2d(
            np.sum(
                dvar * vbeta * zeta * dzeta
                + ddvar
                * vbeta
                * zeta**2
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                + res * vbeta * zeta**2 * dzeta / vtot**2,
                axis=1,
            )[1:]
        )
    if fix_logv_tau or fix_eta:
        h_vt_eta = np.atleast_2d([])
    else:
        h_vt_eta = -np.atleast_2d(
            np.sum(
                dvar * vtau * dzeta * ddzeta
                + ddvar
                * vtau
                * dzeta**2
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                + res * vtau * dzeta**3 / vtot**2,
                axis=1,
            )[1:]
        )

    # Hessian block for (delta_mu, delta_mu)
    if fix_delta_mu:
        h_mu_mu = np.atleast_2d([])
    else:
        a_array = (
            1 / vtot
            + 4 * vbeta * zeta * res / vtot**2
            + vbeta * dvar
            + 2 * vbeta**2 * zeta**2 * ddvar
        )

        b_array = (
            2 * vtau * dzeta * res / vtot**2
            + 2 * vbeta * vtau * zeta * dzeta * ddvar
        )

        c_array = vtau * dvar + 2 * vtau**2 * dzeta**2 * ddvar

        h_mu_mu = np.atleast_2d(
            np.einsum("jpk, jk, jqk", dzeta_dmu, a_array, dzeta_dmu)
            + np.einsum("jpk, jk, jqk", dzeta_dmu, b_array, ddzeta_dmu)
            + np.einsum("jpk, jk, jqk", ddzeta_dmu, b_array, dzeta_dmu)
            + np.einsum("jpk, jk, jqk", ddzeta_dmu, c_array, ddzeta_dmu)
        )

    # Hessian block for (delta_mu, delta_a)
    if fix_delta_mu or fix_delta_a:
        h_mu_a = np.atleast_2d([])
    else:
        a_array = (
            2 * dvar * vbeta * zeta
            + 2 * ddvar * vbeta * zeta * (vbeta * zeta**2 + vtau * dzeta**2)
            + 2 * res * (vbeta * zeta**2 + vtau * dzeta**2) / vtot**2
            + (zeta - res) / vtot
            + 2 * res * vbeta * zeta**2 / vtot**2
        )[1:, :]

        b_array = (
            2
            * vtau
            * dzeta
            * (
                dvar
                + ddvar * (vbeta * zeta**2 + vtau * dzeta**2)
                + res * zeta / vtot**2
            )
        )[1:, :]

        h_mu_a = (
            np.einsum("qj, qpj -> pq", a_array, dzeta_dmu[1:, :, :])
            + np.einsum("qj, qpj -> pq", b_array, ddzeta_dmu[1:, :, :])
        ) / a[np.newaxis, 1:]

    # Hessian block for (delta_mu, eta)
    if fix_delta_mu or fix_eta:
        h_mu_eta = np.atleast_2d([])
    else:
        a_array = (
            dvar * vbeta * dzeta
            + 2
            * ddvar
            * vbeta
            * zeta
            * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
            + 2
            * res
            * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
            / vtot**2
            + dzeta / vtot
            + 2 * vbeta * res * zeta * dzeta / vtot**2
        )[1:, :]

        b_array = (
            dvar * (vbeta * zeta + vtau * ddzeta)
            + 2
            * ddvar
            * vtau
            * dzeta
            * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
            - res / vtot
            + 2 * vtau * res * dzeta**2 / vtot**2
        )[1:, :]

        c_array = (dvar * vtau * dzeta)[1:, :]

        h_mu_eta = -(
            np.einsum("qj, qpj -> pq", a_array, dzeta_dmu[1:, :, :])
            + np.einsum("qj, qpj -> pq", b_array, ddzeta_dmu[1:, :, :])
            + np.einsum("qj, qpj -> pq", c_array, dddzeta_dmu[1:, :, :])
        )

    # Hessian block for (delta_a, delta_a)
    if fix_delta_a:
        h_a_a = np.atleast_2d([])
    else:
        h_a_a = np.diag(
            np.sum(
                2
                * ddvar[1:, :]
                * (vbeta * zeta[1:, :] ** 2 + vtau * dzeta[1:, :] ** 2) ** 2
                + dvar[1:, :]
                * (vbeta * zeta[1:, :] ** 2 + vtau * dzeta[1:, :] ** 2)
                + 4
                * res[1:, :]
                * zeta[1:, :]
                * (vbeta * zeta[1:, :] ** 2 + vtau * dzeta[1:, :] ** 2)
                / vtot[1:, :] ** 2
                + zeta[1:, :] ** 2 / vtot[1:, :],
                axis=1,
            )
            / a[1:] ** 2
        )

    # Hessian block for (delta_a, eta)
    if fix_delta_a or fix_eta:
        h_a_eta = np.atleast_2d([])
    else:
        h_a_eta = -np.diag(
            np.sum(
                2 * dvar * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                + 2
                * ddvar
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                * (vbeta * zeta**2 + vtau * dzeta**2)
                + 2
                * res
                * zeta
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                / vtot**2
                + (zeta - res) * dzeta / vtot
                + 2
                * res
                * dzeta
                * (vbeta * zeta**2 + vtau * dzeta**2)
                / vtot**2,
                axis=1,
            )[1:]
            / a[1:]
        )

    # Hessian block for (eta, eta)
    if fix_eta:
        h_eta_eta = np.atleast_2d([])
    else:
        h_eta_eta = np.diag(
            np.sum(
                dvar
                * (
                    vbeta * (dzeta**2 + zeta * ddzeta)
                    + vtau * (ddzeta**2 + dzeta * dddzeta)
                )
                + 2
                * ddvar
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta) ** 2
                + 4
                * res
                * dzeta
                * (vbeta * zeta * dzeta + vtau * dzeta * ddzeta)
                / vtot**2
                + (dzeta**2 - res * ddzeta) / vtot,
                axis=1,
            )[1:]
        )

    # Boolean array used to compose full Hessian from multiple blocks
    fix = np.array(
        [
            fix_logv_alpha,
            fix_logv_beta,
            fix_logv_tau,
            fix_delta_mu,
            fix_delta_a,
            fix_eta,
        ]
    )

    # Arrange blocks in an object array to enable boolean indexing
    hess_block = np.array(
        [
            [h_va_va, h_va_vb, h_va_vt, h_va_mu, h_va_a, h_va_eta],
            [h_va_vb, h_vb_vb, h_vb_vt, h_vb_mu, h_vb_a, h_vb_eta],
            [h_va_vt.T, h_vb_vt.T, h_vt_vt, h_vt_mu, h_vt_a, h_vt_eta],
            [h_va_mu.T, h_vb_mu.T, h_vt_mu.T, h_mu_mu, h_mu_a, h_mu_eta],
            [h_va_a.T, h_vb_a.T, h_vt_a.T, h_mu_a.T, h_a_a, h_a_eta],
            [
                h_va_eta.T,
                h_vb_eta.T,
                h_vt_eta.T,
                h_mu_eta.T,
                h_a_eta.T,
                h_eta_eta,
            ],
        ],
        dtype=object,
    )

    # Create an index array to select blocks
    idx = np.ix_(~fix, ~fix)

    # Compose the full Jacobian from the selected blocks
    h = np.block(hess_block[idx].tolist())

    # Arrange scale arrays in an object array, then use the fix array to select
    # which arrays to include
    scale_block = np.array(
        [
            [scale_logv_alpha],
            [scale_logv_beta],
            [scale_logv_tau],
            -scale_delta_mu,
            scale_delta_a,
            scale_eta_on_dt,
        ],
        dtype=object,
    )
    scale = np.concatenate(scale_block[~fix].tolist())

    # Return Hessian in scaled internal variables
    return np.diag(scale) @ h @ np.diag(scale)


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
    scale_logv_alpha: float | None = None,
    scale_logv_beta: float | None = None,
    scale_logv_tau: float | None = None,
    scale_delta_mu: ArrayLike | None = None,
    scale_delta_a: ArrayLike | None = None,
    scale_eta: ArrayLike | None = None,
    min_options: dict | None = None,
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
    scale_logv_alpha, scale_logv_beta, scale_logv_tau : float, optional
        Scale for varying the log of the variance parameters.
    scale_delta_mu : array_like with shape(n,), optional
        Scale for varying signal vector. When set to ``None``, the default,
        use ``NoiseModel(sigma_alpha0, sigma_beta0,
        sigma_tau0).noise_amp(mu0)``.
    scale_delta_a : array_like with shape(n,), optional
        Scale for varying signal amplitude drift vector. Default is
        ``np.max((sigma_min, sigma_beta0))`` for all entries, where
        ``sigma_min = np.sqrt(np.min(np.var(x, 1, ddof=1)))``.
    scale_eta : array_like with shape(n,), optional
        Scale for varying signal delay drift vector. Default is
        ``np.max((sigma_min, sigma_tau0))``, for all entries, where
        ``sigma_min = np.sqrt(np.min(np.var(x, 1, ddof=1)))``.
    min_options : dict or None, optional
        Keyword options passed to the ``options`` parameter of
        :func:`scipy.optimize.minimize`. See the documentation on the
        `BFGS <https://docs.scipy.org/doc/scipy/reference/
        optimize.minimize-bfgs.html#optimize-minimize-bfgs>`_
        method for details. By default, ``gtol=1e-5 * x.size``. The options
        ``eps`` and ``finite_diff_rel_step`` are not used.

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
        = \frac{1}{2}\sum_{k=0}^{N-1}\sum_{l=0}^{M-1}& \
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

    >>> import numpy as np
    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> rng = np.random.default_rng(0)
    >>> n, m, dt = 256, 50, 0.05
    >>> t = thz.timebase(n, dt=dt)
    >>> mu = thz.wave(n, dt=dt)
    >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
    >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
    ...  sigma_tau=tau, dt=dt)
    >>> a = 1.0 + 1e-2 * np.concatenate(([0.0],
    ...                                 rng.standard_normal(m - 1)))
    >>> eta = 1e-3 * np.concatenate(([0.0], rng.standard_normal(m - 1)))
    >>> z = thz.scaleshift(np.repeat(np.atleast_2d(mu), m, axis=0), dt=dt,
    ...                    a=a, eta=eta).T  # Orient the array columnwise
    >>> x = z + noise_model.noise_sim(z, axis=0, seed=12345)
    >>> noise_res = thz.noisefit(x, sigma_alpha0=alpha, sigma_beta0=beta,
    ...  sigma_tau0=tau, dt=dt)
    >>> noise_res.noise_model
    NoiseModel(sigma_alpha=0.000100..., sigma_beta=0.00984...,
    sigma_tau=0.000899..., dt=0.05)

    >>> plt.plot(t, np.std(thz.scaleshift(x, a=1 / noise_res.a,
    ... eta=-noise_res.eta, axis=0), axis=1), "-",
    ... label="Data")
    >>> plt.plot(t, noise_res.noise_model.noise_amp(noise_res.mu),
    ...  "--", label="Fit")
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
        scale_logv_alpha=scale_logv_alpha,
        scale_logv_beta=scale_logv_beta,
        scale_logv_tau=scale_logv_tau,
        scale_delta_mu=scale_delta_mu,
        scale_delta_a=scale_delta_a,
        scale_eta=scale_eta,
    )

    objective, jac, x0, input_parsed = parsed

    # Minimize cost function with respect to free parameters
    out = minimize(
        objective,
        x0,
        method="BFGS",
        jac=jac,
        tol=1e-5 * x.size,
        options=min_options,
    )

    return _parse_noisefit_output(out, x, dt=dt, **input_parsed)


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
    scale_logv_alpha: float | None,
    scale_logv_beta: float | None,
    scale_logv_tau: float | None,
    scale_delta_mu: ArrayLike | None,
    scale_delta_a: ArrayLike | None,
    scale_eta: ArrayLike | None,
) -> tuple[Callable, Callable, NDArray[np.float64], dict]:
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
        w = 2 * pi * np.fft.rfftfreq(n, dt)
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

    noise_model = NoiseModel(
        float(sigma_alpha0),
        float(sigma_beta0),
        float(sigma_tau0),
        dt=float(dt),
    )

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
        scale_eta = scale_eta_amp * np.ones(m - 1)

    scale_eta = np.asarray(scale_eta, dtype=np.float64)

    if scale_logv_alpha is None:
        scale_logv_alpha = 1 / m

    if scale_logv_beta is None:
        scale_logv_beta = 1 / m

    if scale_logv_tau is None:
        scale_logv_tau = 1 / m

    scale_logv = np.array(
        [scale_logv_alpha, scale_logv_beta, scale_logv_tau],
        dtype=np.float64,
    )

    # Replace log(x) with -1e2 when x <= 0
    v0 = (
        np.asarray(
            [sigma_alpha0, sigma_beta0, sigma_tau0 / dt], dtype=np.float64
        )
    ) ** 2
    logv0_scaled = np.ma.log(v0).filled(-1.0e2) / scale_logv
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

    eta_scaled0 = eta0[1:] / scale_eta

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
        x0 = np.concatenate((x0, eta_scaled0))

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

        _eta = eta_scaled0 if fix_eta else _p[: m - 1]

        return _nll_noisefit(
            x.T,
            _logv_alpha,
            _logv_beta,
            _logv_tau,
            _delta,
            _epsilon,
            _eta,
            scale_logv_alpha=scale_logv_alpha,
            scale_logv_beta=scale_logv_beta,
            scale_logv_tau=scale_logv_tau,
            scale_delta_mu=scale_delta_mu,
            scale_delta_a=scale_delta_a,
            scale_eta_on_dt=scale_eta / dt,  # Scale in units of dt
        )

    # Bundle free parameters together into objective function
    def jac(_p):
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

        _eta_on_dt = eta_scaled0 / dt if fix_eta else _p[: m - 1]

        return _jac_noisefit(
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
            scale_logv_alpha=scale_logv_alpha,
            scale_logv_beta=scale_logv_beta,
            scale_logv_tau=scale_logv_tau,
            scale_delta_mu=scale_delta_mu,
            scale_delta_a=scale_delta_a,
            scale_eta_on_dt=scale_eta / dt,  # Scale in units of dt
        )

    def hess(_p):
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

        _eta_on_dt = eta_scaled0 / dt if fix_eta else _p[: m - 1]

        return _hess_noisefit(
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
            scale_logv_alpha=scale_logv_alpha,
            scale_logv_beta=scale_logv_beta,
            scale_logv_tau=scale_logv_tau,
            scale_delta_mu=scale_delta_mu,
            scale_delta_a=scale_delta_a,
            scale_eta_on_dt=scale_eta / dt,  # Scale in units of dt
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
        "scale_logv_alpha": scale_logv_alpha,
        "scale_logv_beta": scale_logv_beta,
        "scale_logv_tau": scale_logv_tau,
        "scale_delta_mu": scale_delta_mu,
        "scale_delta_a": scale_delta_a,
        "scale_eta": scale_eta,
        "hess": hess,
    }
    return objective, jac, x0, input_parsed


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
    scale_logv_alpha: float,
    scale_logv_beta: float,
    scale_logv_tau: float,
    scale_delta_mu: NDArray[np.float64],
    scale_delta_a: NDArray[np.float64],
    scale_eta: NDArray[np.float64],
    hess: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NoiseResult:
    """Parse noisefit output"""
    # Parse output
    n, m = x.shape
    bias_correction = np.sqrt(m / (m - 1))

    x_out = out.x
    if fix_sigma_alpha:
        alpha = sigma_alpha0
    else:
        alpha = np.exp(x_out[0] * scale_logv_alpha / 2) * bias_correction
        x_out = x_out[1:]
    if fix_sigma_beta:
        beta = sigma_beta0
    else:
        beta = np.exp(x_out[0] * scale_logv_beta / 2) * bias_correction
        x_out = x_out[1:]
    if fix_sigma_tau:
        tau = sigma_tau0
    else:
        tau = (
            float(np.exp(x_out[0] * scale_logv_tau / 2) * dt) * bias_correction
        )
        x_out = x_out[1:]

    noise_model = NoiseModel(
        float(alpha), float(beta), float(tau), dt=float(dt)
    )

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
        eta_out = np.concatenate(([0.0], x_out[: m - 1] * scale_eta))

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
                    [scale_logv_alpha * alpha / 2],
                    [scale_logv_beta * beta / 2],
                    [scale_logv_tau * tau / 2],
                    scale_delta_mu,
                    scale_delta_a,
                    scale_eta,
                ],
            )
            if not tf
        ]
    )

    # Get or compute the inverse hessian
    hess_inv_scaled = np.linalg.inv(hess(out.x))

    # Convert inverse Hessian into unscaled parameters
    hess_inv = (
        np.diag(scale_hess_inv) @ hess_inv_scaled @ np.diag(scale_hess_inv)
    )

    # Determine parameter uncertainty vector from diagonal entries
    err = np.sqrt(np.diag(hess_inv))
    err_mu = np.array([])
    err_a = np.array([])
    err_eta = np.array([])

    # Parse error vector
    # Propagate error from log(V) to sigma
    if not fix_sigma_alpha:
        err_sigma_alpha = err[0]
        err = err[1:]
    else:
        err_sigma_alpha = 0.0

    if not fix_sigma_beta:
        err_sigma_beta = err[0]
        err = err[1:]
    else:
        err_sigma_beta = 0.0

    if not fix_sigma_tau:
        err_sigma_tau = err[0]
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

    # Cast fun as a Python float in case it is a NumPy constant
    return NoiseResult(
        noise_model,
        mu_out,
        a_out,
        eta_out,
        float(fun),
        hess_inv,
        err_sigma_alpha,
        err_sigma_beta,
        err_sigma_tau,
        err_mu,
        err_a,
        err_eta,
        diagnostic,
    )


@dataclass
class FitResult:
    r"""
    Dataclass for the output of :func:`fit`.

    Parameters
    ----------
    p_opt : ndarray
        Optimal fit parameters.
    p_err : ndarray
        Uncertainty estimate for ``p_opt``,
        ``p_err = np.sqrt(np.diag(p_cov))``.
    p_cov : ndarray
        Covariance matrix estimate for ``p_opt``, determined from the curvature
        of the cost function at ``(p_opt, mu_opt)``.
    mu_opt : ndarray
        Optimal estimate of the input waveform.
    mu_err : ndarray
        Estimated uncertainty in ``mu_opt``, determined from the curvature of
        the cost function at ``(p_opt, mu_opt)``.
    psi_opt : ndarray
        Optimal estimate of the output waveform.
    frfun_opt : complex ndarray
        Estimated values of the frequency response function at non-negative
        frequencies.
    resnorm : float
        Euclidean norm (i.e., sum of the squares) of the normalized total
        least-squares residuals.
    dof : int
        Number of statistical degrees of freedom,
        ``dof = n - n_p - n_a - n_b``, where ``n`` is the number of samples in
        each waveform, ``n_p`` is the number of fit parameters in the frequency
        response function, and ``n_a + n_b`` is the number of real parameters
        necessary to specify the frequency response function at the excluded
        frequencies.
    delta : ndarray
        Residuals of the input waveform ``x``, defined as ``x - mu_opt``.
    epsilon : ndarray
        Residuals of the output waveform ``y``, defined as ``y - psi_opt``,
        where ``psi_opt = thztools.apply_frf(frfun, mu, dt=dt, args=p_opt)``,
        ``frfun`` is the parameterized frequency response function, and
        ``p_opt`` is the array of optimized parameters.
    r_tls : ndarray
        Normalized total least-squares residuals.
    success : bool
        True if one of the convergence criteria is satisfied.
    diagnostic : scipy.optimize.OptimizeResult
        Instance of :class:`scipy.optimize.OptimizeResult` returned by
        :func:`scipy.optimize.least_squares`.

    See Also
    --------
    fit : Fit a frequency response function to time-domain data.
    """

    p_opt: NDArray[np.float64]
    p_err: NDArray[np.float64]
    p_cov: NDArray[np.float64]
    mu_opt: NDArray[np.float64]
    mu_err: NDArray[np.float64]
    psi_opt: NDArray[np.float64]
    frfun_opt: NDArray[np.complex128]
    resnorm: float
    dof: int
    delta: NDArray[np.float64]
    epsilon: NDArray[np.float64]
    r_tls: NDArray[np.float64]
    success: bool
    diagnostic: OptimizeResult


def _costfuntls(
    frfun: Callable,
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
    frfun : callable
        Frequency response function.

            ``frfun(w, *p, *args, **kwargs) -> ndarray``

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

    psi = apply_frf(frfun, mu, dt=dt, args=theta)

    delta_norm = (x - mu) / sigma_x
    eps_norm = (y - psi) / sigma_y

    return np.concatenate((delta_norm, eps_norm))


def fit(
    frfun: Callable,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike,
    noise_parms: ArrayLike | None = None,
    *,
    dt: float | None = None,
    numpy_sign_convention: bool = True,
    args: tuple = (),
    kwargs: dict | None = None,
    f_bounds: ArrayLike | None = None,
    p_bounds: ArrayLike | None = None,
    jac: Callable | None = None,
    lsq_options: dict | None = None,
) -> FitResult:
    r"""
    Fit a parameterized frequency response function to time-domain data.

    Determines the total least-squares fit to ``xdata`` and ``ydata``, given
    the parameterized frequency response function relationship ``frfun`` and
    noise model parameters ``noise_parms``.

    Parameters
    ----------
    frfun : callable
        Frequency response function with signature ``frfun(omega, *p, *args,
        **kwargs)`` that returns an ``ndarray``. Assumes the
        :math:`+i\omega t` convention for harmonic time dependence when
        ``numpy_sign_convention`` is ``True``, the default. All elements of
        ``p`` and ``args`` and all values of ``kwargs`` must be real
        quantities.
    xdata : array_like
        Measured input signal.
    ydata : array_like
        Measured output signal.
    p0 : array_like
        Initial guess for the parameters ``p``.
    noise_parms : array_like or None, optional
        Noise parameters ``(sigma_alpha, sigma_beta, sigma_tau)`` used to
        define the :class:``NoiseModel`` for the fit. Default is ``None``,
        which sets ``noise_parms`` to ``(1.0, 0.0, 0.0)``.
    dt : float or None, optional
        Sampling time, normally in picoseconds. Default is ``None``, which sets
        ``dt`` to ``thztools.options.sampling_time``. If both ``dt`` and
        ``thztools.options.sampling_time`` are ``None``, ``dt`` is set to
        ``1.0``. In this case, the angular frequency ``omega`` must be given in
        units of radians per sampling time, and any parameters in ``args`` must
        be expressed with ``dt`` as the unit of time.
    numpy_sign_convention : bool, optional
        Adopt NumPy sign convention for harmonic time dependence, e.g, express
        a harmonic function with frequency :math:`\omega` as
        :math:`x(t) = a e^{i\omega t}`. Default is ``True``. When set to
        ``False``, uses the convention more common in physics,
        :math:`x(t) = a e^{-i\omega t}`.
    args : tuple
        Additional arguments for ``frfun``. All elements must be real.
    kwargs : dict or None, optional
        Additional keyword arguments for ``frfun``. Default is ``None``, which
        passes no keyword arguments. All values must be real.
    f_bounds : array_like, optional
        Lower and upper bounds on the frequencies included in the fit. For
        ``f_bounds = (f_lo, f_hi)``, the included frequencies ``f`` satisfy

            ``f_lo <= f <= f_hi``.

        The default is ``None``, which sets ``f_bounds`` to include all
        frequencies except zero:

            ``f_bounds = (np.finfo(np.float64).smallest_normal, np.inf)``.

        Set the lower bound to ``0.0`` or any negative number to include zero
        frequency.
    p_bounds : None, 2-tuple of array_like, or :class:`scipy.optimize.Bounds`, optional
        Lower and upper bounds on fit parameter(s). Default is ``None``, which
        sets all parameter bounds to ``(-np.inf, np.inf)``. Each bound
        constraint ``(p_lo, p_hi)`` represents a closed interval for the
        parameter ``p``,

            ``p_lo <= p <= p_hi``.

    jac : callable or None, optional
        Jacobian of the frequency response function with respect to the fit
        parameters, with signature ``jac(w, *p, *args, **kwargs)``. Default is
        ``None``, which uses :func:`scipy.optimize.approx_fprime`.
    lsq_options : dict or None, optional
        Keyword options passed to :func:`scipy.optimize.least_squares`. The
        following keywords are allowed:

            * "ftol" : Tolerance for termination by the change of the cost
              function. Default is 1e-8.
            * "xtol" : Tolerance for termination by the change of the
              independent variables. Default is 1e-8.
            * "gtol" : Tolerance for termination by the norm of the gradient.
              Default is 1e-8.
            * "loss" : Determines the loss function. Default is "linear", which
              gives a standard least-squares problem. Alternative settings have
              not been tested and should be considered experimental.
            * "f_scale" : Value of soft margin between inlier and outlier
              residuals. This parameter has no effect with ``loss="linear"``.
              Default is 1.0.
            * "max_nfev" : Maximum number of function evaluations before the
              termination. If None (default), the value is chosen
              automatically.
            * "verbose" : Level of algorithm's verbosity. Default is 0, which
              works silently.

        See the documentation for :func:`scipy.optimize.least_squares` for
        details.

    Returns
    -------
    res : FitResult
        Fit result represented as a :class:`FitResult` object. Important
        attributes are: ``p_opt``, the optimal fit parameter values; ``p_cov``,
        the parameter covariance matrix estimated from fit; ``resnorm``,
        the value of the total least-squares cost function for the fit;
        ``dof``, the number of statistical degrees of freedom in the fit;
        ``r_tls``, and ``success``, which is ``True`` when the fit converges.
        See the *Notes* section below and the :class:`FitResult` documentation
        for more details and for a description of other attributes.

    Raises
    ------
    ValueError
        If ``noise_parms`` does not have 3 elements.

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
    This function computes the maximum-likelihood estimate for the parameters
    :math:`\boldsymbol{\theta}` in the frequency response function model
    :math:`H(\omega; \boldsymbol{\theta})` by minimizing
    the total least-squares cost function

    .. math:: Q_\text{TLS}(\boldsymbol{\theta}) \
        = \sum_{k=0}^{N-1}\left[ \
        \frac{(x_k - \mu_k)^2}{\sigma_{\mathbf{x},k}^2} \
        + \frac{(y_k - \psi_k)^2}{\sigma_{\mathbf{y},k}^2} \
        \right],

    where :math:`\mathbf{x}` is the input signal array, :math:`\mathbf{y}`
    is the output signal array, and :math:`\boldsymbol{\sigma}_{\mathbf{y}}`,
    :math:`\boldsymbol{\sigma}_{\mathbf{y}}` are their associated noise
    amplitudes. The arrays :math:`\boldsymbol{\mu}` and
    :math:`\boldsymbol{\psi}` are the ideal input and output signal arrays,
    respectively, which are related by the constraint

    .. math:: [\mathcal{F}\boldsymbol{\psi}]_k = \
        [H(\omega_k; \boldsymbol{\theta})] \
        [\mathcal{F}\boldsymbol{\mu}]_k

    at all discrete Fourier transform frequencies :math:`\omega_k` within
    the frequency bounds.

    The optimization step uses :func:`scipy.optimize.least_squares` with ``p``
    and ``mu`` as free parameters. It minimizes the Euclidean norm of the
    residual vector

        ``np.concat((delta / sigma_x, epsilon / sigma_y)``,

    where

        * ``delta = xdata - mu``,
        * ``epsilon = ydata - psi``,
        * ``sigma_x = NoiseModel(*noise_parms, dt=dt).noise_amp(xdata)``,
        * ``sigma_y = NoiseModel(*noise_parms, dt=dt).noise_amp(ydata)``, and
        * ``psi = thz.apply_frf(frfun(omega, *p, *args, **kwargs), mu,
          dt=dt)``.

    References
    ----------
    .. [1] Laleh Mohtashemi, Paul Westlund, Derek G. Sahota, Graham B. Lea,
        Ian Bushfield, Payam Mousavi, and J. Steven Dodge, "Maximum-
        likelihood parameter estimation in terahertz time-domain
        spectroscopy," Opt. Express **29**, 4912-4926 (2021),
        `<https://doi.org/10.1364/OE.417724>`_.

    Examples
    --------
    >>> import numpy as np
    >>> import thztools as thz
    >>> from matplotlib import pyplot as plt
    >>> n, dt = 256, 0.05
    >>> t = thz.timebase(n, dt=dt)
    >>> mu = thz.wave(n, dt=dt)
    >>> alpha, beta, tau = 1e-4, 1e-2, 1e-3
    >>> noise_model = thz.NoiseModel(sigma_alpha=alpha, sigma_beta=beta,
    ...  sigma_tau=tau, dt=dt)

    >>> def frfun(w, amplitude, delay):
    ...    return amplitude * np.exp(1j * w * delay)
    >>>
    >>> p0 = (0.5, 1.0)
    >>> psi = thz.apply_frf(frfun, mu, dt=dt, args=p0)
    >>> xdata = mu + noise_model.noise_sim(mu, seed=0)
    >>> ydata = psi + noise_model.noise_sim(psi, seed=1)
    >>> result = thz.fit(frfun, xdata, ydata, p0, (alpha, beta, tau), dt=dt)
    >>> result.success
    True
    >>> result.p_opt
    array([0.49..., 1.00...])
    >>> result.resnorm
    198.266...

    >>> _, ax = plt.subplots()
    >>> ax.plot(t, result.r_tls, ".")
    >>> ax.set_xlabel("t (ps)")
    >>> ax.set_ylabel(r"$r_\mathrm{TLS}$")
    >>> plt.show()
    """
    fit_method = "trf"

    args = np.atleast_1d(args)
    if kwargs is None:
        kwargs = {}

    def _frfun_local(
        _omega: NDArray[np.float64], *p: np.float64
    ) -> NDArray[np.complex128]:
        out = frfun(_omega, *p, *args, **kwargs)
        if not numpy_sign_convention:
            return np.conj(out)
        return out

    p0 = np.atleast_1d(p0).astype(np.float64)
    xdata = np.asarray(xdata, dtype=np.float64)
    ydata = np.asarray(ydata, dtype=np.float64)

    if noise_parms is None:
        noise_parms = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    else:
        noise_parms = np.asarray(noise_parms, dtype=np.float64)

    if noise_parms.size != NUM_NOISE_PARAMETERS:
        msg = f"sigma_parms must be a tuple of length {NUM_NOISE_PARAMETERS:d}"
        raise ValueError(msg)

    dt = float(_assign_sampling_time(dt))

    n = ydata.shape[-1]
    n_p = len(p0)

    if f_bounds is None:
        f_bounds = np.array(
            (np.finfo(np.float64).smallest_normal, np.inf), dtype=np.float64
        )
    else:
        f_bounds = np.asarray(f_bounds, dtype=np.float64)

    f = rfftfreq(n, dt)
    f_excl_lo_idx = f < f_bounds[0]
    f_excl_hi_idx = f > f_bounds[1]
    f_incl_idx = ~f_excl_lo_idx * ~f_excl_hi_idx

    w = 2 * pi * f

    n_below = np.sum(f_excl_lo_idx)
    n_in = np.sum(f_incl_idx)
    n_above = np.sum(f_excl_hi_idx)
    n_ex = n_below + n_above

    n_a = 0
    n_b = 0

    if n_ex > 0:
        n_a = n_ex
        n_b = n_ex - 1
        if n % 2 == 0 and n_below > 0 and n_above > 0:
            n_b -= 1
        elif n % 2 == 1 and n_below == 0:
            n_b += 1

    alpha, beta, tau = noise_parms.tolist()
    # noinspection PyArgumentList
    noise_model = NoiseModel(alpha, beta, tau, dt=dt)
    sigma_x = noise_model.noise_amp(xdata)
    sigma_y = noise_model.noise_amp(ydata)
    v_x = np.diag(sigma_x**2)
    v_y = np.diag(sigma_y**2)
    p0_est = np.concatenate((p0, np.zeros(n_a), np.zeros(n_b), np.zeros(n)))

    if p_bounds is None:
        p_bounds = (-np.inf, np.inf)
        fit_method = "lm"
    else:
        p_bounds = np.asarray(p_bounds, dtype=np.float64)
        if p_bounds.shape[0] == 2:  # noqa: PLR2004
            p_bounds = (
                np.concatenate(
                    (p_bounds[0], np.full((n_a + n_b + n,), -np.inf))
                ),
                np.concatenate(
                    (p_bounds[1], np.full((n_a + n_b + n,), np.inf))
                ),
            )
        else:
            msg = "`bounds` must contain 2 elements."
            raise ValueError(msg)

    if lsq_options is None:
        lsq_options = {}
    else:
        valid_keys = {
            "ftol",
            "xtol",
            "gtol",
            "loss",
            "f_scale",
            "max_nfev",
            "verbose",
        }
        bad_keys = set(lsq_options.keys()) - valid_keys
        if len(bad_keys) > 0:
            msg = f"Invalid key(s) {list(bad_keys)} in `lsq_options`"
            raise KeyError(msg)

    def fun_ex(
        _a: NDArray[np.float64], _b: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        # Transfer function is purely real at the first and last frequencies
        if n_a - n_b == 2:  # noqa: PLR2004
            return np.concatenate(([_a[0]], _a[1:-1] + _b * 1j, [_a[-1]]))

        # Transfer function is purely real at either the first or last
        # frequency
        if n_a - n_b == 1:
            # TF is real at last frequency
            if n_below == 0:
                return np.concatenate((_a[:-1] + _b * 1j, [_a[-1]]))

            # Otherwise TF is real at first frequency
            return np.concatenate(([_a[0]], _a[1:] + _b * 1j))

        # Transfer function is complex at all excluded frequencies
        return _a + _b * 1j

    def function(
        _w: NDArray[np.float64], *_theta: np.float64
    ) -> NDArray[np.complex128]:
        _a = np.asarray(_theta[n_p : n_p + n_a], dtype=np.float64)
        _b = np.asarray(_theta[n_p + n_a :], dtype=np.float64)
        h_ex = fun_ex(_a, _b)
        h_in = _frfun_local(_w[f_incl_idx], *_theta[:n_p])
        return np.concatenate((h_ex[:n_below], h_in, h_ex[n_below:]))

    def function_flat(_x: NDArray[np.float64]) -> NDArray[np.float64]:
        _tf = _frfun_local(w[f_incl_idx], *_x)
        return np.concatenate((np.real(_tf), np.imag(_tf)))

    def jacobian_fun(_p: NDArray[np.float64]) -> NDArray[np.complex128]:
        if jac is None:
            # If Jacobian is not supplied, compute it numerically
            _tf_prime = approx_fprime(_p, function_flat)
            _tf_prime_complex = _tf_prime[0:n_in] + 1j * _tf_prime[n_in:]
            out = np.atleast_2d(_tf_prime_complex).T
        else:
            # Otherwise, return supplied Jacobian
            out = np.atleast_2d(jac(w[f_incl_idx], *_p, *args, **kwargs))[
                :, :n_p
            ].T
        if not numpy_sign_convention:
            return np.conj(out)
        return out

    def jacobian_bl(
        _p: NDArray[np.float64], _fft_mu: NDArray[np.complex128]
    ) -> NDArray[np.float64]:
        fft_jac_bl = np.concatenate(
            (
                np.zeros((n_p, n_below)),
                jacobian_fun(_p),
                np.zeros((n_p, n_above)),
            ),
            axis=-1,
        )
        jac_bl = irfft(fft_jac_bl * _fft_mu, n=n)
        if n_a > 0:
            a_circ = la.circulant(signal.unit_impulse(n_a))
            jac_a = np.concatenate(
                (
                    a_circ[:, :n_below],
                    np.zeros((n_a, n_in)),
                    a_circ[:, n_below:],
                ),
                axis=-1,
            )
            jac_bl_a = irfft(jac_a * _fft_mu, n=n)
            jac_bl = np.concatenate((jac_bl, jac_bl_a), axis=0)
        if n_b > 0:
            b_circ = la.circulant(signal.unit_impulse(n_b) * 1j)
            if n_a - n_b == 2:  # noqa: PLR2004
                jac_b = np.concatenate(
                    (
                        np.zeros((n_b, 1)),
                        b_circ[:, : n_below - 1],
                        np.zeros((n_b, n_in)),
                        b_circ[:, n_below - 1 :],
                        np.zeros((n_b, 1)),
                    ),
                    axis=-1,
                )
            elif n_a - n_b == 1:
                if n_below == 0:
                    jac_b = np.concatenate(
                        (
                            np.zeros((n_b, n_in)),
                            b_circ[:, :],
                            np.zeros((n_b, 1)),
                        ),
                        axis=-1,
                    )
                else:
                    jac_b = np.concatenate(
                        (
                            np.zeros((n_b, 1)),
                            b_circ[:, : n_below - 1],
                            np.zeros((n_b, n_in)),
                            b_circ[:, n_below - 1 :],
                        ),
                        axis=-1,
                    )
            else:
                jac_b = np.concatenate(
                    (np.zeros((n_b, n_in)), b_circ[:, :]), axis=-1
                )
            jac_bl_b = irfft(jac_b * _fft_mu, n=n)
            jac_bl = np.concatenate((jac_bl, jac_bl_b), axis=0)
        return jac_bl

    def jac_fun(_x: NDArray[np.float64]) -> NDArray[np.float64]:
        p_est = _x[: n_p + n_a + n_b]
        mu_est = xdata[:] - _x[n_p + n_a + n_b :]
        jac_tl = np.zeros((n, n_p + n_a + n_b))
        jac_tr = np.diag(1 / sigma_x)
        fft_mu_est = rfft(mu_est)
        jac_bl = -(jacobian_bl(p_est[:n_p], fft_mu_est) / sigma_y).T
        impulse_response = apply_frf(
            function, signal.unit_impulse(n), dt=dt, args=p_est
        )
        jac_br = (la.circulant(impulse_response).T / sigma_y).T

        return np.block([[jac_tl, jac_tr], [jac_bl, jac_br]])

    result = opt.least_squares(
        lambda _p: _costfuntls(
            function,
            _p[: n_p + n_a + n_b],
            xdata[:] - _p[n_p + n_a + n_b :],
            xdata[:],
            ydata[:],
            sigma_x[:],
            sigma_y[:],
            dt,
        ),
        p0_est,
        jac=jac_fun,
        bounds=p_bounds,
        method=fit_method,
        x_scale=np.concatenate((np.ones(n_p + n_a + n_b), sigma_x)),
        **lsq_options,
    )

    # Parse output
    _, s, vt = la.svd(result.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    cov = np.dot(vt.T / s**2, vt)

    p_opt_all = result.x[: n_p + n_a + n_b]
    p_opt = result.x[:n_p]
    p_cov = cov[:n_p, :n_p]
    p_err = np.sqrt(np.diag(p_cov))
    delta = result.x[n_p + n_a + n_b :]

    mu_opt = xdata - delta
    mu_err = np.sqrt(np.diag(cov)[n_p + n_a + n_b :])
    psi_opt = apply_frf(function, mu_opt, dt=dt, args=p_opt_all)
    epsilon = ydata - psi_opt
    resnorm = 2 * result.cost
    dof = n - n_p - n_a - n_b

    h_circ = la.circulant(
        apply_frf(function, signal.unit_impulse(n), dt=dt, args=p_opt_all)
    )
    h_delta = apply_frf(function, delta, dt=dt, args=p_opt_all)
    u_x = h_circ @ v_x @ h_circ.T
    r_tls = sqrtm(np.linalg.inv(v_y + u_x)) @ (epsilon - h_delta)

    frfun_opt = function(w, *p_opt_all)

    # Cast resnorm as a Python float and success as a Python bool, in case
    # either is a NumPy constant
    return FitResult(
        p_opt=p_opt,
        p_err=p_err,
        p_cov=p_cov,
        mu_opt=mu_opt,
        mu_err=mu_err,
        psi_opt=psi_opt,
        frfun_opt=frfun_opt,
        resnorm=float(resnorm),
        dof=dof,
        delta=delta,
        epsilon=epsilon,
        r_tls=r_tls,
        success=bool(result.success),
        diagnostic=result,
    )
