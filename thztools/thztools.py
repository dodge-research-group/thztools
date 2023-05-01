import numpy as np
import scipy.linalg


def fftfreq(n, t):
    """
    Computes the positive and negative frequencies sampled in the Fast Fourier Transform.

    Parameters
    ----------
    n : int
        Number of time samples
    t: float
        Sampling time

    Returns
    -------
    f : ndarray
        Frequency vector (1/ts) of length n containing the sample frequencies.

    """

    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n / 2)] = -f[int(n / 2)]

    return f


def noisevar(sigma, mu, t):
    """
 Noisevar computes the time-domain noise variance for noise parameters sigma, with a signal mu and sampling interval T.
 There are three noise parameters: the first corresponds to amplitude noise, in signal units (i.e, the same units as mu)
 ;  the second corresponds to multiplicative noise, which is dimensionless; and the third corresponds to timebase noise,
 in units of signal/time, where the units for time are the same as for T. The output, Vmu, is given in units that are
 the square of the signal units.



 Parameters
 ----------

   sigma : ndarray
    (3, ) array  containing noise parameters

   mu : ndarray
       (n, ) array  containing  signal vector

   t : float
    Sampling time

 Returns
 --------
   vmu : ndarray
        Noise variance


    """

    n = len(mu)
    w = 2 * np.pi * fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j * w * np.fft.fft(mu)))

    return sigma[0] ** 2 + (sigma[1] * mu) ** 2 + (sigma[2] * mudot) ** 2


def pulsegen(n, t0, w, a, t):
    """ Pulsegen generates a short pulse of temporal width w centered at t0 for use in tests of time-domain analysis

    Parameters
    ----------

    n : int
        number of sampled points

    t0 : float
        pulse center

    w : float
        pulse width

    a : float
        pulse amplitude

    t : float
        sampling time


    Returns
    -------

    y : ndarray
        signal (u.a)

    t : float
        timebase

    """

    t2 = t * np.arange(n)
    tt = (t2 - t0) / w

    y = a * (1 - 2 * tt ** 2) * np.exp(-tt ** 2)

    return [y, t2]


def thzgen(n, t, t0):
    """
    Generate a terahertz pulse with n points at sampling interval t and centered at t0.

    Parameters
    ----------

    n : int
        number of sampled points.

    t : float
        sampling time

    t0 : float
        pulse center

    Returns
    -------

    y : ndarray
        signal (u.a)

    t : float
        timebase

    """

    default_a = 1
    default_taur = 0.3
    default_tauc = 0.1
    default_taul = 0.05 / np.sqrt(2 * np.log(2))

    a = default_a
    taur = default_taur
    tauc = default_tauc
    taul = default_taul

    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n / 2)] = -f[int(n / 2)]

    w = 2 * np.pi * f
    l = np.exp(-(w * taul) ** 2 / 2) / np.sqrt(2 * np.pi * taul ** 2)
    r = 1 / (1 / taur - 1j * w) - 1 / (1 / taur + 1 / tauc - 1j * w)
    s = -1j * w * (l * r) ** 2 * np.exp(1j * w * t0)

    t2 = t * np.arange(n)

    y = np.real(np.fft.ifft(np.conj(s)))
    y = a * y / np.max(y)

    return [y, t2]


def costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft):
    """

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        mu : ndarray
            Signal vector of size (n,).

        theta : ndarray
            Input parameters for the function.

        xx : ndarray
            Measured input signal.

        yy : ndarray
            Measured output signal.

        sigmax : ndarray or matrix
            Noise covariance matrix of the input signal.

        sigmay : nadarray
            Noise covariance matrix of the output signal.

        wfft :


    Returns
    -------
    res : callable


    """
    wfft1 = np.array([wfft])
    n = len(sigmax)
    #n =np.shape(wfft)[0]
    h = np.conj(fun(theta, wfft))
    if n % 2 == 0:
        kny = n // 2
        h[kny] = np.real(h[kny])

    ry = yy - np.real(np.fft.ifft(np.fft.fft(xx) * h))
    vy = np.diag(sigmay**2)

    htilde = np.fft.ifft(h)

    uy = np.zeros((n, n))
    for k in np.arange(n):
        a = np.reshape(np.roll(htilde, k), (n, 1))
        b = np.reshape(np.conj(np.roll(htilde, k)), (1, n))
        uy = uy + np.real(np.dot(a, b)) * sigmax[k]**2
        #uy = uy + np.real(np.roll(htilde, k-1) @ np.roll(htilde, k-1).T) @ sigmax[k]**2

    w = np.dot(np.eye(n), scipy.linalg.inv(scipy.linalg.sqrtm(uy + vy)))
    res = np.dot(w, ry)

    return res


def epswater(f, t=25):
    """
    epswater(f,T) computes the complex relative permittivity at frequency f
    (THz) and temperature T (deg C)

    Parameters
    ----------
    f Frequency (THz)
    t Temperature (deg C) (optional)

    Returns
    -------
    epsilonr: complex
     Complex relative permittivity of water

    References
    ----------

    .. [1] Ellison, W. J. (2007). Permittivity of pure water, at standard atmospheric pressure, over the frequency
    range 0–25 THz and the temperature range 0–100 C. Journal of physical and chemical reference data, 36(1), 1-18.

    """
    # Frequency conversion to Hz
    f = f * 1e12

    # Define relaxation parameters
    a = np.array([79.23882, 3.815866, 1.634967])
    b = np.array([0.004300598, 0.01117295, 0.006841548])
    c = np.array([1.382264e-13, 3.510354e-16, 6.30035e-15])
    d = np.array([652.7648, 1249.533, 405.5169])
    tc = 133.1383

    # Define resonance parameters
    p0 = 0.8379692
    p = np.array([-0.006118594, -0.000012936798, 4235901000000, -14260880000,
                  273815700, -1246943, 9.618642e-14, 1.795786e-16, -9.310017e-18,
                  1.655473e-19, 0.6165532, 0.007238532, -0.00009523366, 15983170000000,
                  -74413570000, 497448000, 2.882476e-14, -3.142118e-16, 3.528051e-18])

    # Compute temperature - dependent functions
    eps0 = 87.9144 - 0.404399 * t + 9.58726e-4 * t ** 2 - 1.32802e-6 * t ** 3
    delta = a * np.exp(-b * t)
    tau = c * np.exp(d / (t + tc))

    delta4 = p0 + p[0] * t + p[1] * t ** 2
    f0 = p[2] + p[3] * t + p[4] * t ** 2 + p[5] * t ** 3
    tau4 = p[6] + p[7] * t + p[8] * t ** 2 + p[9] * t ** 3
    delta5 = p[10] + p[11] * t + p[12] * t ** 2
    f1 = p[13] + p[14] * t + p[15] * t ** 2
    tau5 = p[16] + p[17] * t + p[18] * t ** 2

    # Put it all together
    epsilonr = (eps0 + 2 * 1j * np.pi * f * (delta[0] * tau[0] / (1 - 2 * 1j * np.pi * f * tau[0])
                                             + delta[1] * tau[1] / (1 - 2 * 1j * np.pi * f * tau[1])
                                             + delta[2] * tau[2] / (1 - 2 * 1j * np.pi * f * tau[2]))
                + 1j * np.pi * f * (delta4 * tau4 / (1 - 2 * 1j * np.pi * tau4 * (f0 + f))
                                    + delta4 * tau4 / (1 + 2 * 1j * np.pi * tau4 * (f0 - f)))
                + 1j * np.pi * f * (delta5 * tau5 / (1 - 2 * 1j * np.pi * tau5 * (f1 + f))
                                    + delta5 * tau5 / (1 + 2 * 1j * np.pi * tau5 * (f1 - f))))

    return epsilonr


def shiftmtx(tau, n, ts):
    """
    Shiftmtx computes the n by n transfer matrix for a continuous time-shift.

    Parameters
    -----------

    tau : float
        Input parameters for the function

    n : int
        Number of time samples

    ts: int
        sampling time

    Returns
    -------
    h: ndarray or matrix
        (n, n) Transfer matrix

    """

    # Fourier method
    f = fftfreq(n, ts)
    w = 2 * np.pi * f

    imp = np.fft.ifft(np.exp(-1j * w * tau)).real

    # computes the n by n transformation matrix
    h = linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h

def tdtf(fun, theta, n, ts):
    """
    Returns the transfer matrix for a given function.

    It computes the n-by-n transfer matrix for the given function fun with
    input parameter theta. Note that the transfer function should be written
    using the physicist's -iwt convention, and that it should be in the format fun(theta,w), where theta
    is a vector of the function parameters. The transfer function must be
    Hermitian.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        theta : ndarray
            Input parameters for the function.

        n : int
            Number of time samples.

        ts : ndarray
            Sampling time.

    Returns
    -------
        h : ndarray or matrix
            Transfer matrix with size (n,n).

    """

    # compute the transfer function over positive frequencies
    fs = 1 / (ts * n)
    fp = fs * np.arange(0, math.floor((n - 1) / 2 + 1))
    wp = 2 * np.pi * fp
    tfunp = fun(theta, wp)

    # The transfer function is Hermitian, so we evaluate negative frequencies
    # by taking the complex conjugate of the corresponding positive frequency.
    # Include the value of the transfer function at the Nyquist frequency for
    # even n.
    if n % 2 != 0:
        tfun = np.concatenate((tfunp, np.conj(np.flipud(tfunp[1:]))))

    else:
        wny = np.pi * n * fs
        #print('tfunp', tfunp)
        tfun = np.concatenate((tfunp, np.conj(np.concatenate((fun(theta, wny), np.flipud(tfunp[1:]))))))

    # Evaluate the impulse response by taking the inverse Fourier transform,
    # taking the complex conjugate first to convert to ... +iwt convention

    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h

def tdnll(x, param, fix):
    """
    Computes negative log-likelihood for the time-domain noise model.
    Tdnll computes the negative log-likelihood function for obtaining the
    data matrix x, given the parameter dictionary param.
    Parameters
    ----------
    x : ndarray or matrix
        Data matrix
    param : dict
        A dictionary containing parameters including:
        logv : ndarray
            Array of size (3, ) containing log of noise parameters.
        mu : ndarray
            Signal vector of size (n,).
        a: ndarray
            Amplitude vector of size (m,).
        eta : ndarray
            Delay vector of size (m,).
        ts : float
            Sampling time.
        d : ndarray or matrix
            n-by-n derivative matrix.
    fix : dict
        A dictionary containing variables to fix for the gradient calculation.
        logv : bool
            Log of noise parameters.
        mu : bool
            Signal vector.
        a : bool
            Amplitude vector.
        eta : bool
            Delay vector.
    Returns
    -------
    nll : callable
        Negative log-likelihood function
    gradnll : ndarray
        Gradient of the negative log-likelihood function
    """

    # Parse function inputs
    [n, m] = x.shape

    # Parse parameter dictionary
    pfields = param.keys()
    ignore = dict()
    if 'logv' in pfields:
        v = np.exp(param['logv'])
        v = np.reshape(v, (len(v), 1))

    else:
        raise ValueError('Tdnll requires Param structure with logv field')

    if 'mu' in pfields:
        mu = param['mu']
        mu = np.reshape(mu, (len(mu), 1))

    else:
        raise ValueError('Tdnll requires param structure with mu field')
    pass

    if 'a' in pfields and param['a'] != []:
        a = param['a']
        a = np.reshape(a, (len(a), 1))
        ignore['a'] = False
    else:
        a = np.ones((m, 1))
        ignore['a'] = True
    pass

    if 'eta' in pfields and param['eta'] != []:
        eta = param['eta']
        eta = np.reshape(eta, (len(eta), 1))
        ignore['eta'] = False
    else:
        eta = np.zeros((m, 1))
        ignore['eta'] = True
    pass

    if 'ts' in pfields:
        ts = param['ts']
    else:
        ts = 1
        raise ValueError('TDNLL received Param structure without ts field; set to one')
    pass

    if 'd' in pfields:
        d = param['d']
    else:
        # Compute derivative matrix
        def fun(theta, w):
            return - 1j * w
        d = tdtf(fun, 0, n, ts)
    pass

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(n, ts)
    w = 2 * np.pi * f
    w = w.reshape(len(w), 1)
    mu_f = np.fft.fft(mu.flatten()).reshape(len(mu), 1)

    gradcalc = np.logical_not([[fix['logv']], [fix['mu']], [fix['a'] or ignore['a']], [fix['eta'] or ignore['eta']]])

    if ignore['eta']:
        zeta = mu * np.conj(a).T
        zeta_f = np.fft.fft(zeta, axis=0)
    else:
        exp_iweta = np.exp(1j * np.tile(w, m) * np.conj(np.tile(eta, n)).T)
        zeta_f = np.conj(np.tile(a, n)).T * np.conj(exp_iweta) * np.tile(mu_f, m)
        zeta = np.real(np.fft.ifft(zeta_f, axis=0))

    # Compute negative - log likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Simplest case: just variance and signal parameters, A and eta fixed at
    # defaults
    if ignore['a'] and ignore['eta']:
        dmu = np.real(np.fft.ifft(1j * w * mu_f, axis=0))
        valpha = v[0]
        vbeta = v[1] * mu**2
        vtau = v[2] * dmu ** 2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / np.tile(vtot, m)
        nll = m * n * np.log(2 * np.pi) / 2 + (m / 2) * np.sum(np.log(vtot)) + np.sum(resnormsq) / 2

        # Compute gradient if requested
        # if nargout > 1:
        ngrad = np.sum(gradcalc[0:2] * [[3], [n]])
        gradnll = np.zeros((ngrad, 1))
        nstart = 0
        dvar = (vtot - np.mean(ressq, axis=1).reshape(n, 1)) / vtot ** 2
        if gradcalc[0]:
            gradnll[nstart] = (m / 2) * np.sum(dvar) * v[0]
            gradnll[nstart + 1] = (m / 2) * np.sum(mu ** 2 * dvar) * v[1]
            gradnll[nstart + 2] = (m / 2) * np.sum(dmu ** 2. * dvar) * v[2]
            nstart = nstart + 3
        if gradcalc[1]:
            print('mu shape : ', mu.shape)
            print('dvar shape: ', dvar.shape)
            print('d shape: ', d.shape)
            print('Dmu shape: ', dmu.shape)
            gradnll[nstart:nstart + n] = m * (v[1] * mu * dvar + v[2] * np.dot(d.T, (dmu * dvar)) - np.mean(res, axis=1)
                                              .reshape(n, 1) / vtot)

    # Alternative case: A, eta, or both are not set to defaults
    else:
        dzeta = np.real(np.fft.ifft(1j * np.tile(w, m) * zeta_f, axis=0))

        valpha = v[0]
        vbeta = v[1] * zeta**2
        vtau = v[2] * dzeta ** 2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / vtot
        nll = m * n * np.log(2 * np.pi) / 2 + np.sum(np.log(vtot)) / 2 + np.sum(resnormsq) / 2

        # Compute gradient if requested
        # if nargout > 1:
        ngrad = np.sum(gradcalc * [[3], [n], [m], [m]])
        gradnll = np.zeros((ngrad, 1))
        nstart = 0
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot**2
        if gradcalc[0]:
            # Gradient wrt logv
            gradnll[nstart] = (1 / 2) * np.sum(dvar) * v[0]
            gradnll[nstart + 1] = (1 / 2) * np.sum(zeta.flatten() ** 2 * dvar.flatten()) * v[1]
            gradnll[nstart + 2] = (1 / 2) * np.sum(dzeta.flatten() ** 2 * dvar.flatten()) * v[2]
            nstart = nstart + 3
        if gradcalc[1]:
            # Gradient wrt mu
            p = np.fft.fft(v[1] * dvar * zeta - reswt, axis=0) - 1j * v[2] * w * np.fft.fft(dvar * dzeta, axis=0)
            gradnll[nstart:nstart + n] = np.sum(np.conj(a).T * np.real(np.fft.ifft(exp_iweta * p, axis=0)), axis=1).reshape(n, 1)
            nstart = nstart + n
        if gradcalc[2]:
            # Gradient wrt A
            term = (vtot - valpha) * dvar - reswt * zeta
            if np.any(np.isclose(a, 0)):
                raise ValueError("One or more elements of the amplitude vector are close to zero ")
            gradnll[nstart:nstart + m] = np.conj(np.sum(term, axis=0)).reshape(m, 1) / a
            if not fix['mu']:
                gradnll = np.delete(gradnll, nstart)
                nstart = nstart + m - 1
            else:
                nstart = nstart + m
        if gradcalc[3]:
            # Gradient wrt eta
            ddzeta = np.real(np.fft.ifft(-np.tile(w, m) ** 2 * zeta_f, axis=0))
            gradnll = np.squeeze(gradnll)
            gradnll[nstart:nstart + m] = -np.sum(dvar * (zeta * dzeta * v[1] + dzeta * ddzeta * v[2]) - reswt * dzeta,
                                                 axis=0).reshape(m, )

            if not fix['mu']:
                gradnll = np.delete(gradnll, nstart)
    gradnll = gradnll.flatten()

    return nll, gradnll

def tdnoisefit(x, param, fix={'logv': False, 'mu': False, 'a': True, 'eta': True}, ignore={'a': True, 'eta': True}):
    """ Computes maximum likelihood estimates parameters for the time-domain noise model.
     Tdnoisefit computes the noise parameters sigma and the underlying signal
     vector mu for the data matrix x, where the columns of x are each noisy
     measurements of mu.
    Parameters
    ----------
    x : ndarray or matrix
        Data matrix.
    param : dict
        A dictionary containing initial guess for the parameters:
        v0 : ndarray
            Initial guess, noise model parameters. Array of real elements of size (3, )
        mu0 : ndarray
            Initial guess, signal vector. Array of real elements of size  (n, )
        a0 : ndarray
            Initial guess, amplitude vector. Array of real elements of size (m, )
        eta0 : ndarray
            Initial guess, delay vector. Array of real elements of size (m, )
        ts : int
            Sampling time
    fix : dict, optional
        A dictionary containing variables to fix for the minimization.
        logv : bool
            Log of noise parameters.
        mu : bool
            Signal vector.
        a : bool
            Amplitude vector.
        eta : bool
            Delay vector.
        If not given, chosen to set free all the variables.
    ignore : dict
        A dictionary containing variables to ignore for the minimization.
        a : bool
            Amplitude vector.
        eta : bool
            Delay vector.
        If not given, chosen to ignore both amplitude and delay.
    Returns
    --------
    p : dict
        Output parameter dictionary containing:
            eta : ndarray
                Delay vector.
            a : ndarray
                Amplitude vector.
            mu : ndarray
                Signal vector.
            var : ndarray
                Log of noise parameters
        fval : float
           Value of NLL cost function from FMINUNC
        Diagnostic : dict
            Dictionary containing diagnostic information
                err : dic
                    Dictionary containing  error of the parameters.
                grad : ndarray
                      Negative loglikelihood cost function gradient from scipy.optimize.minimize BFGS method.
                hessian : ndarray
                    Negative loglikelihood cost function hessian from scipy.optimize.minimize BFGS method.
     """
    n, m = x.shape

    # Parse Inputs
    if 'v0' in param:
        v0 = param['v0']
    else:
        v0 = np.mean(np.var(x, 1)) * np.array([1, 1, 1])
        param['v0'] = v0

    if 'mu0' in param:
        mu0 = param['mu0']
    else:
        mu0 = np.mean(x, 1)
        param['mu0'] = mu0

    if 'a0' in param:
        a0 = param['a0']
    else:
        a0 = np.ones(m)
        param['a0'] = a0

    if 'eta0' in param:
        eta0 = param['eta0']

    else:
        eta0 = np.zeros(m)
        param['eta0'] = eta0

    if 'ts' in param:
        ts = param['ts']
    else:
        ts = 1
        param['ts'] = 1

    mle = {'x0': np.array([])}
    idxstart = 0
    idxrange = dict()

    # If fix['logv'], return log(v0); otherwise return logv parameters
    if fix['logv']:
        def setplogv(p):
            return np.log(param['v0'])
    else:
        mle['x0'] = np.concatenate((mle['x0'], np.log(param['v0'])))
        idxend = idxstart + 3
        idxrange['logv'] = np.arange(idxstart, idxend)

        def setplogv(p):
            return p[idxrange['logv']]

        idxstart = idxend
    pass

    # If Fix['mu'], return mu0, otherwise, return mu parameters
    if fix['mu']:
        def setpmu(p):
            return param['mu0']
    else:
        mle['x0'] = np.concatenate((mle['x0'], param['mu0']))
        idxend = idxstart + n
        idxrange['mu'] = np.arange(idxstart, idxend)

        def setpmu(p):
            return p[idxrange['mu']]

        idxstart = idxend
    pass

    # If ignore A, return []; if Fix['A'], return A0; if ~Fix.A & Fix.mu, return all A parameters;
    # If ~Fix.A & ~Fix.mu, return all A parameters but first

    if ignore['a']:
        def setpa(p):
            return []

    elif fix['a']:
        def setpa(p):
            return param['a0']

    elif fix['mu']:
        mle['x0'] = np.concatenate((mle['x0'], param['a0']))
        idxend = idxstart + m
        idxrange['a'] = np.arange(idxstart, idxend)

        def setpa(p):
            return p[idxrange['a']]

        idxstart = idxend
    else:
        mle['x0'] = np.concatenate((mle['x0'], param['a0'][1:] / param['a0'][0]))
        idxend = idxstart + m - 1
        idxrange['a'] = np.arange(idxstart, idxend)

        def setpa(p):
            return np.concatenate(([1], p[idxrange['a']]), axis=0)

        idxstart = idxend
    pass

    # If Ignore.eta, return []; if Fix.eta, return eta0; if ~Fix.eta & Fix.mu,return all eta parameters;
    # if ~Fix.eta & ~Fix.mu, return all eta parameters but first

    if ignore['eta']:
        def setpeta(p):
            return []

    elif fix['eta']:
        def setpeta(p):
            return param['eta0']

    elif fix['mu']:
        mle['x0'] = np.concatenate((mle['x0'], param['eta0']))
        idxend = idxstart + m
        idxrange['eta'] = np.arange(idxstart, idxend)

        def setpeta(p):
            return p[idxrange['eta']]

    else:
        mle['x0'] = np.concatenate((mle['x0'], param['eta0'][1:] - param['eta0'][0]))
        idxend = idxstart + m - 1
        idxrange['eta'] = np.arange(idxstart, idxend)

        def setpeta(p):
            return np.concatenate(([0], p[idxrange['eta']]), axis=0)
    pass

    def fun(theta, w):
        return -1j * w

    d = tdtf(fun, 0, n, param['ts'])

    def parsein(p):
        return {'logv': setplogv(p), 'mu': setpmu(p), 'a': setpa(p), 'eta': setpeta(p), 'ts': param['ts'], 'd': d}

    def objective(p):
        return tdnll(x, parsein(p), fix)[0]

    def jacobian(p):
        return tdnll(x, parsein(p), fix)[1]

    mle['objective'] = objective
    out = minimize(mle['objective'], mle['x0'], method='BFGS', jac=jacobian)

    # The trust-region algorithm returns the Hessian for the next-to-last
    # iterate, which may not be near the final point. To check, test for
    # positive definiteness by attempting to Cholesky factorize it. If it
    # returns an error, rerun the optimization with the quasi-Newton algorithm
    # from the current optimal point.

    try:
        np.linalg.cholesky(np.linalg.inv(out.hess_inv))
        hess = np.linalg.inv(out.hess_inv)
    except:
        print('Hessian returned by FMINUNC is not positive definite;\n'
              'recalculating with quasi-Newton algorithm')

        mle['x0'] = out.x
        out2 = minimize(mle['objective'], mle['x0'], method='BFGS', jac=jacobian)
        hess = np.linalg.inv(out2.hess_inv)

    # Parse output
    p = {}
    idxrange = dict()
    idxstart = 0

    if fix['logv']:
        p['var'] = param['v0']
    else:
        idxend = idxstart + 3
        idxrange['logv'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['var'] = np.exp(out.x[idxrange['logv']])
    pass

    if fix['mu']:
        p['mu'] = param['mu0']
    else:
        idxend = idxstart + n
        idxrange['mu'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['mu'] = out.x[idxrange['mu']]
    pass

    if ignore['a'] or fix['a']:
        p['a'] = param['a0']
    elif fix['mu']:
        idxend = idxstart + m
        idxrange['a'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['a'] = out.x[idxrange['a']]
    else:
        idxend = idxstart + m - 1
        idxrange['a'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['a'] = np.concatenate(([1], out.x[idxrange['a']]), axis=0)
    pass

    if ignore['eta'] or fix['eta']:
        p['eta'] = param['eta0']
    elif fix['mu']:
        idxend = idxstart + m
        idxrange['eta'] = np.arange(idxstart, idxend)
        p['eta'] = out.x[idxrange['eta']]
    else:
        idxend = idxstart + m - 1
        idxrange['eta'] = np.arange(idxstart, idxend)
        p['eta'] = np.concatenate(([0], out.x[idxrange['eta']]), axis=0)
    pass

    p['ts'] = param['ts']

    varyParam = np.logical_not([fix['logv'], fix['mu'], fix['a'] or ignore['a'], fix['eta'] or ignore['eta']])
    diagnostic = {'grad': out.jac, 'hessian': hess,
                  'err': {'var': [], 'mu': [], 'a': [], 'eta': []}}
    v = np.dot(np.eye(hess.shape[0]), scipy.linalg.inv(hess))
    err = np.sqrt(np.diag(v))
    idxstart = 0
    if varyParam[0]:
        diagnostic['err']['var'] = np.sqrt(np.diag(np.diag(p['var']) * v[0:3, 0:3]) * np.diag(p['var']))
        idxstart = idxstart + 3
    pass

    if varyParam[1]:
        diagnostic['err']['mu'] = err[idxstart:idxstart + n]
        idxstart = idxstart + n
    pass

    if varyParam[2]:
        if varyParam[1]:
            diagnostic['err']['a'] = err[idxstart:idxstart + m - 1]
            idxstart = idxstart + m - 1
        else:
            diagnostic['err']['a'] = err[idxstart:idxstart + m]
            idxstart = idxstart + m
    pass

    if varyParam[3]:
        if varyParam[1]:
            diagnostic['err']['eta'] = err[idxstart:idxstart + m - 1]
        else:
            diagnostic['err']['eta'] = err[idxstart:idxstart + m]
    pass

    return [p, out.fun, diagnostic]