import numpy as np

def thzgen(N, T, t0, varargin):
    default_A = 1
    default_taur = 0.3
    default_tauc = 0.1
    default_taul = 0.05 / np.sqrt(2 * np.log(2))

    A = default_A
    taur = default_taur
    tauc = default_tauc
    taul = default_taul

    if(N % 2 == 1):
        f = np.fft.fftfreq(N, T)
    else:
        f = np.fft.fftfreq(N, T)
        f[int(N/2)] = -f[int(N/2)]


    w = 2 * np.pi * f

    L = np.exp(-(w * taul)**2 / 2) / np.sqrt(2 * np.pi * taul**2);
    R = 1/(1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w);
    S = -1j * w * (L * R)**2 * np.exp(1j * w * t0);

    t = T*np.arange(N)
    # t = t(:);

    y = np.real(   np.fft.ifft(  np.conj(S) )  )
    y = A * y / np.max(y)

    return [y,t]