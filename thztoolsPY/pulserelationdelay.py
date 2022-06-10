import math


def pulserelationdelay(signalIn, Parms):
    # N = len(signalIn)
    N = np.shape(signalIn)[0]
    c = np.shape(signalIn)[1]
    Nmax = math.floor((N - 1) / 2)
    nTerm = Parms.shape[0]

    if c = 1:
        signalIn = signalIn[:, 0]

    else c != 1:
        print('Warning: Multiple colums in data arrays, operating on first')
        signalIn = signalIn[:, 0]

    # Fourier transform signal
    signalInft = np.fft.fft(signalIn)
    tFun = np.zeros(Nmax)

    for i in range(nTerm + 1):
        na = len(Parms[0]) - 1
        nb = len(Parms[1]) - 1

        nab = np.max(na, nb)

        for i in range(nab + 1):
            vanderS = ()