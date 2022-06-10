import numpy as np


def fftfreq(n, t):
    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n/2)] = -f[int(n/2)]

    return f
