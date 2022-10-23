import numpy as np


def fftfreq(n, t):
    """
    Computes the positive and negative frequencies sampled in the Fast Fourier Transform.

    Parameters
    ----------
    n : int
        Number of time samples
    t: float
        Sampling time

    Returns
    -------
    f : ndarray
        Frequency vector (1/ts) of length n containing the sample frequencies.

    """

    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n/2)] = -f[int(n/2)]

    return f
