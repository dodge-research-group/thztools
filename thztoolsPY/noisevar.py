import numpy as np

from thztoolsPY.fftfreq import fftfreq


def noisevar(sigma, mu, t):

    n = len(mu)
    w = 2*np.pi*fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j*w*np.fft.fft(mu)))

    return sigma[0]**2 + (sigma[1]*mu)**2 + (sigma[2]*mudot)**2
