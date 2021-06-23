import numpy as np
from numpy.fft import fft, ifft


def fftfreq(n, ts):
    """
    Generates array frequencies for the discrete Fourier transform
    Parameters
    ----------
    n: int
      Number of points in the array.
    ts: float
      Sampling time.

    Returns
    -------
    array_like float

    Notes
    -----
    The convention used here is different from the NumPy fft module when n is even. Here,
    the Nyquist frequency is positive, and in the NumPy fft module, the Nyquist frequency
    is negative.
    """
    x = np.arange(n)
    xsub = x - np.floor((n - 1) / 2)
    kcirc = np.roll(xsub, np.ceil((n + 1) / 2).astype(int))
    f = kcirc / (n * ts)

    return f


def noisevar(sigma, mu, ts):
    """
    Compute the time-domain noise variance.
    Parameters
    ----------
    sigma: array_like float
        Array with 3 elements.
        sigma[0]: Additive noise amplitude (units of mu)
        sigma[1]: Multiplicative noise amplitude (dimensionless)
        sigma[2]: Timebase noise amplitude (units of ts)
    mu: array_like float
        Signal array.
    ts: float
        Sampling time.

    Returns
    -------
    array_like float
        Noise variance for each element of mu.
    """
    # Use FFT to compute derivative of mu for timebase noise
    n = int(len(mu))
    w = 2 * np.pi * fftfreq(n, ts)
    muf = fft(mu)
    mudotifft = ifft(1j * w * muf)
    mudot = np.real(mudotifft)

    # Compute noise variance
    cov_mu = sigma[0] ** 2 + (sigma[1] * mu)**2 + (sigma[2] * mudot)**2
    return cov_mu
