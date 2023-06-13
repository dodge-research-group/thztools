from __future__ import annotations

import warnings
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import scipy.linalg  # type: ignore
from numpy.fft import irfft, rfft, rfftfreq
from numpy.typing import ArrayLike
from scipy.optimize import minimize  # type: ignore


class Wave:
    r"""
    Signal vector with associated information.

    Attributes
    ==========
    signal : ndarray, optional
        Signal vector. Default is an empty array.

    ts : float, optional
        Sampling time in ps. Default is 1.0.

    t0 : float, optional
        Absolute time associated with first data point. Default is 0.0.

    metadata : dict, optional
        Dictionary of metadata associated with the wave object. Default is an
        empty dictionary.
    """
    def __init__(
            self,
            signal: ArrayLike = None,
            ts: float = 1.0,
            t0: float = 0.0,
            metadata: dict = None,
    ) -> None:
        if signal is None:
            self.signal = np.array([])
        else:
            self.signal = np.asarray(signal)
        self.ts = ts
        self.t0 = t0
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

    def __repr__(self):
        return (f"{self.__class__.__name__}(signal={self.signal.__repr__()}, "
                f"ts={self.ts}, t0={self.t0}, metadata="
                f"{self.metadata.__repr__()})")

    def __array__(self):
        # See
        # https://numpy.org/doc/stable/user/basics.dispatch.html#basics
        # -dispatch
        # for details on NumPy custom array containers
        return self.signal

    @property
    def t(self) -> ArrayLike:
        r"""
        Generate array of sampled times.

        Returns
        -------
        ndarray
            Array of sampled times associated with the signal, beginning with
            t0 and separated by ts.
        """
        return

    @property
    def f(self) -> ArrayLike:
        r"""
        Generate array of sampled frequencies.

        Returns
        -------
        ndarray
            Array of frequencies associated with the signal. Generate with
            numpy.rfftfreq.

        """
        return

    @property
    def spectrum(self) -> ArrayLike:
        r"""
        Complex spectrum of signal.

        Returns
        -------
        ndarray
            Real Fourier transform of signal. Generate with numpy.rfft

        """
        return

    @property
    def psd(self) -> ArrayLike:
        r"""
        Power spectral density of signal.

        Returns
        -------
        ndarray
            Power spectral density of signal. Generate with
            scipy.signal.periodogram. Some of the optional parameters may be
            useful to include, such as 'window', 'detrend', and 'scaling'.

        """
        return

    def load(self, filepath: str) -> None:
        r"""
        Load ``Wave`` object from a data file.

        Parameters
        ----------
        filepath : str
            File path to read.

        Returns
        -------

        """
        return


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
    ell = np.exp(-((w * taul) ** 2) / 2) / np.sqrt(2 * np.pi * taul ** 2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (ell * r) ** 2 * np.exp(1j * w * t0)

    t2 = ts * np.arange(n)

    y = irfft(np.conj(s), n=n)
    y = a * y / np.max(y)

    return y, t2


class DataPulse:
    def __init__(self, filename=""):
        self.AcquisitionTime = None
        self.Description = None
        self.TimeConstant = None
        self.WaitTime = None
        self.setPoint = None
        self.scanOffset = None
        self.temperature = None
        self.time = None
        self.amplitude = None
        self.ChannelAMaximum = None
        self.ChannelAMinimum = None
        self.ChannelAVariance = None
        self.ChannelASlope = None
        self.ChannelAOffset = None
        self.ChannelBMaximum = None
        self.ChannelBMinimum = None
        self.ChannelBVariance = None
        self.ChannelBSlope = None
        self.ChannelBOffset = None
        self.ChannelCMaximum = None
        self.ChannelCMinimum = None
        self.ChannelCVariance = None
        self.ChannelCSlope = None
        self.ChannelCOffset = None
        self.ChannelDMaximum = None
        self.ChannelDMinimum = None
        self.ChannelDVariance = None
        self.ChannelDSlope = None
        self.ChannelDOffset = None
        self.dirname = None
        self.file = None
        self.filename = filename
        self.frequency = None

        if filename is not None:
            data = pd.read_csv(filename, header=None, delimiter="\t")

            ind = 0
            for i in range(data.shape[0]):
                try:
                    float(data[0][i])
                except ValueError:
                    ind = i + 1
                pass

            keys = data[0][0:ind].to_list()
            vals = data[1][0:ind].to_list()

            for k in range(len(keys)):
                if keys[k] in list(self.__dict__.keys()):
                    try:
                        float(vals[k])
                        setattr(self, keys[k], float(vals[k]))
                    except ValueError:
                        setattr(self, keys[k], vals[k])

                else:
                    msg = "The data key is not defied"
                    raise Warning(msg)

            self.time = data[0][ind:].to_numpy(dtype=float)
            self.amplitude = data[1][ind:].to_numpy(dtype=float)

            # Calculate frequency range
            self.frequency = (
                                     np.arange(
                                         np.floor(len(self.time))) / 2 - 1
                             ).T / (self.time[-1] - self.time[0])

            # Calculate fft
            famp = np.fft.fft(self.amplitude)
            self.famp = famp[0: int(np.floor(len(famp) / 2))]


def shiftmtx(tau: float, n: int, ts: float = 1) -> ArrayLike:
    """
    Shiftmtx computes the n by n transfer matrix for a continuous time-shift.

    Parameters
    -----------

    tau : float
        Delay.

    n : int
        Number of samples.

    ts: float, optional
        Sampling time.

    Returns
    -------
    h: ndarray
        Transfer matrix with shape (n, n).

    """

    # Fourier method
    f = rfftfreq(n, ts)
    w = 2 * np.pi * f

    imp = irfft(np.exp(-1j * w * tau), n=n)

    # computes the n by n transformation matrix
    h = scipy.linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h


def airscancorrect(
        x: ArrayLike,
        *,
        a: ArrayLike | None = None,
        eta: ArrayLike | None = None,
        ts: float = 1.0,
) -> ArrayLike:
    """Rescales and shifts each column of the matrix x.

    Parameters
    ----------
    x : array_like
        Data array with shape (n, m).
    a : array_like, optional
        Amplitude correction. If set, a must have shape (m,). Otherwise, no
        correction is applied.
    eta : array_like, optional
        Delay correction. If set, a must have shape (m,). Otherwise, no
        correction is applied.
    ts : float, optional
        Sampling time. Default is 1.0.

    Returns
    -------
    xadj : ndarray
        Adjusted data array.

    """
    x = np.asarray(x)

    [n, m] = x.shape

    if a is None:
        a = np.ones((m,))
    else:
        a = np.asarray(a)

    if eta is None:
        eta = np.zeros((m,))
    else:
        eta = np.asarray(eta)

    xadj = np.zeros((n, m))
    # TODO: refactor with broadcasting
    for i in np.arange(m):
        s = shiftmtx(-eta[i], n, ts)
        xadj[:, i] = s @ (x[:, i] / a[i])

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
    r"""Computes the maximum likelihood cost function.

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
            Noise covariance matrix of the input signal.

        sigmay : array_like
            Noise covariance matrix of the output signal.

        ts : float
            Sampling time.

    Returns
    -------
    res : array_like


    """
    n = xx.shape[0]
    wfft = 2 * np.pi * rfftfreq(n, ts)
    h = np.conj(fun(theta, wfft))

    ry = yy - irfft(rfft(xx) * h, n=n)
    vy = np.diag(sigmay ** 2)

    htilde = irfft(h, n=n)

    uy = np.zeros((n, n))
    for k in np.arange(n):
        a = np.reshape(np.roll(htilde, k), (n, 1))
        b = np.reshape(np.conj(np.roll(htilde, k)), (1, n))
        uy = uy + np.real(np.dot(a, b)) * sigmax[k] ** 2
        # uy = uy + np.real(np.roll(htilde, k-1) @ np.roll(htilde, k-1).T) @
        # sigmax[k]**2

    w = np.dot(np.eye(n), scipy.linalg.inv(scipy.linalg.sqrtm(uy + vy)))
    res = np.dot(w, ry)

    return res


def tdtf(fun: Callable, theta: ArrayLike, n: int, ts: float) -> ArrayLike:
    """
    Computes the time-domain transfer matrix for a frequency response function.

    Parameters
    ----------
        fun : callable
            Frequency function, in the form fun(theta, w), where theta
            is a vector of the function parameters. The function should be
            expressed in the -iwt convention and must be Hermitian.

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

    # compute the transfer function over positive frequencies
    if not isinstance(ts, np.ndarray):
        ts = np.array([ts])
    else:
        ts = ts

    fs = 1 / (ts * n)
    fp = fs * np.arange(0, (n - 1) // 2 + 1)
    wp = 2 * np.pi * fp
    tfunp = fun(theta, wp)

    # The transfer function is Hermitian, so we evaluate negative frequencies
    # by taking the complex conjugate of the corresponding positive frequency.
    # Include the value of the transfer function at the Nyquist frequency for
    # even n.
    if n % 2 != 0:
        tfun = np.concatenate((tfunp, np.conj(np.flipud(tfunp[1:]))))

    else:
        wny = np.pi * n * fs
        # print('tfunp', tfunp)
        tfun = np.concatenate(
            (
                tfunp,
                np.conj(
                    np.concatenate((fun(theta, wny), np.flipud(tfunp[1:])))
                ),
            )
        )

    # Evaluate the impulse response by taking the inverse Fourier transform,
    # taking the complex conjugate first to convert to ... +iwt convention

    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = scipy.linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h


def tdnll(
        x: ArrayLike,
        logv: ArrayLike,
        mu: ArrayLike,
        a: ArrayLike | None = None,
        eta: ArrayLike | None = None,
        ts: float = 1.0,
        d: ArrayLike | None = None,
        fix_logv: bool = False,
        fix_mu: bool = False,
        fix_a: bool = False,
        fix_eta: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    r"""
    Compute negative log-likelihood for the time-domain noise model.

    Computes the negative log-likelihood function for obtaining the
    data matrix ``x``, given the parameter dictionary param.

    Parameters
    ----------
    x : ndarray or matrix
        Data matrix
    logv : ndarray
        Array of size (3, ) containing log of noise parameters.
    mu : ndarray
        Signal vector of size (n,).
    a: ndarray, optional
        Amplitude vector of size (m,).
    eta : ndarray, optional
        Delay vector of size (m,).
    ts : float, optional
        Sampling time.
    d : ndarray, optional
        Derivative matrix, size (n, n).
    fix_logv : bool, optional
        Log of noise parameters.
    fix_mu : bool, optional
        Signal vector.
    fix_a : bool, optional
        Amplitude vector.
    fix_eta : bool, optional
        Delay vector.

    Returns
    -------
    nll : callable
        Negative log-likelihood function
    gradnll : ndarray
        Gradient of the negative log-likelihood function
    """
    # Parameters to ignore when computing gradnll
    ignore_a = False
    ignore_eta = False

    # Parse and validate function inputs
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Data array x must be 2D.")
    n, m = x.shape

    logv = np.asarray(logv)
    if logv.size != 3:
        raise ValueError("Noise parameter array logv must have 3 elements.")

    mu = np.asarray(mu)
    if mu.ndim != 1:
        raise ValueError("Ideal signal vector mu must be 1D.")
    if mu.size != n:
        raise ValueError("Size of mu is incompatible with data array x.")
    mu = np.reshape(mu, (n, 1))

    if a is None:
        a = np.ones((m,))
        ignore_a = True
    else:
        a = np.asarray(a)
        if a.size != m:
            raise ValueError("Size of a is incompatible with data array x.")
    a = np.reshape(a, (m, 1))

    if eta is None:
        eta = np.ones((m,))
        ignore_eta = True
    else:
        eta = np.asarray(eta)
        if eta.size != m:
            raise ValueError("Size of eta is incompatible with data array x.")
    eta = np.reshape(eta, (m, 1))

    if d is None:
        def fun(_, _w):
            return -1j * _w

        d = tdtf(fun, 0, n, ts)

    # Compute variance
    v = np.exp(logv)
    v = np.reshape(v, (len(v), 1))

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(n, ts)
    w = 2 * np.pi * f
    w = w.reshape(len(w), 1)
    mu_f = np.fft.fft(mu.flatten()).reshape(len(mu), 1)

    gradcalc = np.logical_not(
        [
            [fix_logv],
            [fix_mu],
            [fix_a or ignore_a],
            [fix_eta or ignore_eta],
        ]
    )

    if ignore_eta:
        zeta = mu * np.conj(a).T
        zeta_f = np.fft.fft(zeta, axis=0)
    else:
        exp_iweta = np.exp(1j * np.tile(w, m) * np.conj(np.tile(eta, n)).T)
        zeta_f = (
                np.conj(np.tile(a, n)).T * np.conj(exp_iweta) * np.tile(mu_f,
                                                                        m)
        )
        zeta = np.real(np.fft.ifft(zeta_f, axis=0))

    # Compute negative - log likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res ** 2

    # Simplest case: just variance and signal parameters, A and eta fixed at
    # defaults
    if ignore_a and ignore_eta:
        dmu = np.real(np.fft.ifft(1j * w * mu_f, axis=0))
        valpha = v[0]
        vbeta = v[1] * mu ** 2
        vtau = v[2] * dmu ** 2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / np.tile(vtot, m)
        nll = (
                m * n * np.log(2 * np.pi) / 2
                + (m / 2) * np.sum(np.log(vtot))
                + np.sum(resnormsq) / 2
        )

        # Compute gradient if requested
        # if nargout > 1:
        ngrad = np.sum(gradcalc[0:2] * [[3], [n]])
        gradnll = np.zeros((ngrad, 1))
        nstart = 0
        dvar = (vtot - np.mean(ressq, axis=1).reshape(n, 1)) / vtot ** 2
        if gradcalc[0]:
            gradnll[nstart] = (m / 2) * np.sum(dvar) * v[0]
            gradnll[nstart + 1] = (m / 2) * np.sum(mu ** 2 * dvar) * v[1]
            gradnll[nstart + 2] = (m / 2) * np.sum(dmu ** 2.0 * dvar) * v[2]
            nstart = nstart + 3
        if gradcalc[1]:
            # print('mu shape : ', mu.shape)
            # print('dvar shape: ', dvar.shape)
            # print('d shape: ', d.shape)
            # print('Dmu shape: ', dmu.shape)
            gradnll[nstart: nstart + n] = m * (
                    v[1] * mu * dvar
                    + v[2] * np.dot(d.T, (dmu * dvar))
                    - np.mean(res, axis=1).reshape(n, 1) / vtot
            )

    # Alternative case: A, eta, or both are not set to defaults
    else:
        dzeta = np.real(np.fft.ifft(1j * np.tile(w, m) * zeta_f, axis=0))

        valpha = v[0]
        vbeta = v[1] * zeta ** 2
        vtau = v[2] * dzeta ** 2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / vtot
        nll = (
                m * n * np.log(2 * np.pi) / 2
                + np.sum(np.log(vtot)) / 2
                + np.sum(resnormsq) / 2
        )

        # Compute gradient if requested
        # if nargout > 1:
        ngrad = np.sum(gradcalc * [[3], [n], [m], [m]])
        gradnll = np.zeros((ngrad, 1))
        nstart = 0
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot ** 2
        if gradcalc[0]:
            # Gradient wrt logv
            gradnll[nstart] = 0.5 * np.sum(dvar) * v[0]
            gradnll[nstart + 1] = (
                    0.5 * np.sum(zeta.flatten() ** 2 * dvar.flatten()) * v[1]
            )
            gradnll[nstart + 2] = (
                    0.5 * np.sum(dzeta.flatten() ** 2 * dvar.flatten()) * v[2]
            )
            nstart = nstart + 3
        if gradcalc[1]:
            # Gradient wrt mu
            p = np.fft.fft(v[1] * dvar * zeta - reswt, axis=0) - 1j * v[
                2
            ] * w * np.fft.fft(dvar * dzeta, axis=0)
            gradnll[nstart: nstart + n] = np.sum(
                np.conj(a).T * np.real(np.fft.ifft(exp_iweta * p, axis=0)),
                axis=1,
            ).reshape(n, 1)
            nstart = nstart + n
        if gradcalc[2]:
            # Gradient wrt A
            term = (vtot - valpha) * dvar - reswt * zeta
            if np.any(np.isclose(a, 0)):
                msg = (
                    "One or more elements of the amplitude vector are "
                    "close to zero "
                )
                raise ValueError(msg)
            gradnll[nstart: nstart + m] = (
                    np.conj(np.sum(term, axis=0)).reshape(m, 1) / a
            )
            if not fix_mu:
                gradnll = np.delete(gradnll, nstart)
                nstart = nstart + m - 1
            else:
                nstart = nstart + m
        if gradcalc[3]:
            # Gradient wrt eta
            ddzeta = np.real(np.fft.ifft(-np.tile(w, m) ** 2 * zeta_f, axis=0))
            gradnll = np.squeeze(gradnll)
            gradnll[nstart: nstart + m] = -np.sum(
                dvar * (zeta * dzeta * v[1] + dzeta * ddzeta * v[2])
                - reswt * dzeta,
                axis=0,
            ).reshape(
                m,
            )

            if not fix_mu:
                gradnll = np.delete(gradnll, nstart)
    gradnll = gradnll.flatten()

    return nll, gradnll


def tdnoisefit(
        x: ArrayLike,
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
        ignore_eta: bool = True
) -> Tuple[dict, float, dict]:
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
        raise ValueError("Data array x must be 2D.")
    n, m = x.shape

    if v0 is None:
        v0 = np.mean(np.var(x, 1)) * np.array([1, 1, 1])
    else:
        v0 = np.asarray(v0)
        if v0.size != 3:
            raise ValueError(
                "Noise parameter array logv must have 3 elements."
            )

    if mu0 is None:
        mu0 = np.mean(x, 1)
    else:
        mu0 = np.asarray(mu0)
        if mu0.size != n:
            raise ValueError("Size of mu0 is incompatible with data array x.")

    if a0 is None:
        a0 = np.ones(m)
    else:
        a0 = np.asarray(a0)
        if a0.size != m:
            raise ValueError("Size of a0 is incompatible with data array x.")

    if eta0 is None:
        eta0 = np.ones(m)
    else:
        eta0 = np.asarray(eta0)
        if eta0.size != m:
            raise ValueError("Size of eta0 is incompatible with data array x.")

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
        mle["x0"] = np.concatenate(
            (mle["x0"], a0[1:] / a0[0])
        )
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
        mle["x0"] = np.concatenate(
            (mle["x0"], eta0[1:] - eta0[0])
        )
        idxend = idxstart + m - 1
        idxrange["eta"] = np.arange(idxstart, idxend)

        def setpeta(_p):
            return np.concatenate(([0], _p[idxrange["eta"]]), axis=0)

    pass

    def fun(_, _w):
        return -1j * _w

    d = tdtf(fun, 0, n, ts)

    def parsein(_p):
        return {
            "logv": setplogv(_p),
            "mu": setpmu(_p),
            "a": setpa(_p),
            "eta": setpeta(_p),
            "ts": ts,
            "d": d,
        }

    def objective(_p):
        return tdnll(x, *parsein(_p).values(),
                     fix_v, fix_mu, fix_a, fix_eta)[0]

    def jacobian(_p):
        return tdnll(x, *parsein(_p).values(),
                     fix_v, fix_mu, fix_a, fix_eta)[1]

    mle["objective"] = objective
    out = minimize(mle["objective"], mle["x0"], method="BFGS", jac=jacobian)

    # The trust-region algorithm returns the Hessian for the next-to-last
    # iterate, which may not be near the final point. To check, test for
    # positive definiteness by attempting to Cholesky factorize it. If it
    # returns an error, rerun the optimization with the quasi-Newton algorithm
    # from the current optimal point.

    try:
        np.linalg.cholesky(np.linalg.inv(out.hess_inv))
        hess = np.linalg.inv(out.hess_inv)
    except np.linalg.LinAlgError:
        print(
            "Hessian returned by FMINUNC is not positive definite;\n"
            "recalculating with quasi-Newton algorithm"
        )

        mle["x0"] = out.x
        out2 = minimize(
            mle["objective"], mle["x0"], method="BFGS", jac=jacobian
        )
        hess = np.linalg.inv(out2.hess_inv)

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
    v = np.dot(np.eye(hess.shape[0]), scipy.linalg.inv(hess))
    err = np.sqrt(np.diag(v))
    idxstart = 0
    if vary_param[0]:
        diagnostic["err"]["var"] = np.sqrt(
            np.diag(np.diag(p["var"]) * v[0:3, 0:3]) * np.diag(p["var"])
        )
        idxstart = idxstart + 3
    pass

    if vary_param[1]:
        diagnostic["err"]["mu"] = err[idxstart: idxstart + n]
        idxstart = idxstart + n
    pass

    if vary_param[2]:
        if vary_param[1]:
            diagnostic["err"]["a"] = err[idxstart: idxstart + m - 1]
            idxstart = idxstart + m - 1
        else:
            diagnostic["err"]["a"] = err[idxstart: idxstart + m]
            idxstart = idxstart + m
    pass

    if vary_param[3]:
        if vary_param[1]:
            diagnostic["err"]["eta"] = err[idxstart: idxstart + m - 1]
        else:
            diagnostic["err"]["eta"] = err[idxstart: idxstart + m]
    pass

    return [p, out.fun, diagnostic]
