import math
import numpy as np


def pulserelationdelay(signalin, parms):
    n = len(signalin)
    nmax = math.floor((n - 1) / 2)
    # nTerm = len(parms[0][0])

    w = np.ones(nmax)

    for i in range(len(w)):
        f = (i + 1) / n
        w[i] = 2 * np.pi * f
    s = 1j*w

    # Fourier transform signal
    signalinft = np.fft.fft(signalin)
    iin = signalinft[2:nmax + 1]

    for i in range(len(parms[0])):
        na = len(parms[0][i])
    for j in range(len(parms[0])):
        nb = len(parms[1][j])

    nab = max(na, nb)

    kval = np.zeros(nab)
    for k in range(len(kval)):
        kval[k] = k

    vanders = np.power(np.tile((np.reshape(w, (len(w), 1))), nab),
                       kval)  # first re shape w, expands it and then exponentiation

    denominator = vanders @ parms[0][0]
    numerator = vanders @ parms[1][0]

    tfun = np.zeros(nmax)

    for m in range(len(parms[2])):
        tfun = tfun + (numerator / denominator) * np.exp(parms[2][m] * s)

    out = tfun * iin

    if n % 2 == 0:
        signalout = np.real(np.fft.ifft(np.concatenate((np.array([0]), out, np.array([0]), np.conj(np.flipud(out))))))

    else:
        signalout = np.real(np.fft.ifft(np.concatenate((np.array([0]), out, np.conj(np.flipud(out))))))

    return signalout
