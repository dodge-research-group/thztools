def pulserelationdelay(signalIn, Parmsa, Parmsb):
    N = len(signalIn)
    Nmax = math.floor((N - 1) / 2)
    nTerm = len(Parms[0][0])

    w = np.ones(Nmax)
    for i in range(len(w)):
      f = (i+1)/N
      w[i] = 2*np.pi*f
      s = 1j*w


    # Fourier transform signal
    signalInFt = np.fft.fft(signalIn)
    In = signalInFt[2:Nmax+1]

    for i in range(len(Parms[0])):
      na = len(Parms[0][i])
    for j in range(len(Parms[1])):
      nb = len(Parms[1][j])


    nab = max(na, nb)

    kVal = np.zeros(nab)
    for i in range(len(kVal)):
      kVal[i] = i

    vanderS = np.power(np.tile((np.reshape(w, (len(w), 1))), nab), kVal) # first re shape w, expands it and then exponentiation

    denominator = vanderS@Parms[0][0]
    numerator = vanderS@Parms[1][0]


    tFun = np.zeros(Nmax)

    for i in range(len(Parms[2])):
      tFun = tFun + (numerator/denominator)*np.exp(Parms[2][i]*s)

    out = tFun*In

    if N % 2== 0:
      signalOut = np.real(np.fft.ifft(np.concatenate((np.array([0]), out, np.array([0]), np.conj(np.flipud(out))))))

    else:
      signalOut = np.real(np.fft.ifft(np.concatenate((np.array([0]), out, np.conj(np.flipud(out))))))


    return signalOut