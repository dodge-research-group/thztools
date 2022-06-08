import numpy as np

def fftfreq(N,T):
    if(N % 2 == 1):
        f = np.fft.fftfreq(N, T)
    else:
        f = np.fft.fftfreq(N, T)
        f[int(N/2)] = -f[int(N/2)]

    return f