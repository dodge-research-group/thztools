import numpy as np
from scipy import linalg
import math


def tdtf(fun, theta, n, ts):
    """
    Returns the transfer matrix for a given function.

    It computes the n-by-n transfer matrix for the given function fun with
    input parameter theta. Note that the transfer function should be written
    using the physicist's -iwt convention, and that it should be in the format fun(theta,w), where theta
    is a vector of the function parameters. The transfer function must be
    Hermitian.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        theta : ndarray
            Input parameters for the function.

        n : int
            Number of time samples.

        ts : float
            Sampling time.

    Returns
    -------
        h : ndarray or matrix
            Transfer matrix with size (n,n).

    """

    # compute the transfer function over positive frequencies
    fs = 1 / (ts * n)
    fp = fs * np.arange(0, math.floor((n - 1) / 2 + 1))
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
        tfun = np.concatenate((tfunp, np.conj(np.concatenate((fun(theta, wny), np.flipud(tfunp[1:]))))))

    # Evaluate the impulse response by taking the inverse Fourier transform,
    # taking the complex conjugate first to convert to ... +iwt convention

    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h
