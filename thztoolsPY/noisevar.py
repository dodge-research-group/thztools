import numpy as np

from thztools.thztoolsPY.fftfreq import fftfreq


def noisevar(sigma, mu, T):

    N = len(mu)
    w = 2*np.pi*fftfreq(N, T)
    mudot = np.real(np.fft.ifft(1j*w*np.fft.fft(mu)))

    return sigma[0]**2 + (sigma[1]*mu)**2 + (sigma[2]*mudot)**2