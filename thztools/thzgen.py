import numpy as np
import thztools
from scipy import constants
 


'''
THZGEN generates a terahertz pulse with N points, at sampling interval T, centered at t0

Inputs:
    N     number of sampled points
    T     sampling time
    t0    pulse center

Outputs:
    y     signal
    t     timebase [T]
'''


def thzgen(N, T, t0):

    f = thztools.fftfreq(N, ts)
    w = 2 * np.pi * f

    # Computing L
    Lsq = np.square(w * taul * -1)  # (-(w*taul).^2/2)

    L = np.exp(Lsq / 2) / np.sqrt(2 * np.pi * (taul ** 2))

    # computing R

    iw = -1j * w
    invw = 1 / iw

    R = 1 / (1 / taur - iw) - 1 / (1 / taur + 1 / tauc - iw)

    # computing S

    S = iw * (L * R) ** 2 * np.exp(iw * t0)

    #  computing timebase matrix by multiplying sample number and the sampling time
    #  first create a matrix with N elements, then add one so that indexing begins at 1

    tm = np.arange(13)
    tm = tm + 1
    t = T * tm

    # y calcuations

    Sconj = np.conj(S)
    Sifft = np.fft.ifft(Sconj)
    y = np.real(Sifft)
    y = A * y / max(y)

    return y

 
