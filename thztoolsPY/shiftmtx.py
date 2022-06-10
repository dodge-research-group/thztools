import numpy as np
from scipy import linalg

from thztoolsPY.fftfreq import fftfreq


def shiftmtx(tau, n, ts):
    """
    shiftmtx computes the n by n transfer matrix for a continuous time-shift

        parameters:
            tau: Input parameters for the function
            n: Number of time samples
            ts: sampling time

        returns:
            H: Transfer matrix with size (n,n)
    """

    # Fourier method
    f = fftfreq(n, ts)
    w = 2*np.pi*f

    imp = np.fft.ifft(np.exp(-1j*w*tau)).real

    # computes the n by n transformation matrix
    h = linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h
