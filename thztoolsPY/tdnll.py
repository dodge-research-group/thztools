import numpy as np
from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.tdtf import tdtf


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