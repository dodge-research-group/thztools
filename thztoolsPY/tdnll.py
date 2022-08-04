import numpy as np

from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.tdtf import tdtf


def tdnll(x, param, Fix):
    """
    TDNLL computes negative log-likelihood for the time-domain noise model

    Syntax:   nll = tdnll(x,Param,Fix)

    Description:

    TDNLL computes the negative log-likelihood function for obtaining the
    data matrix x, given the parameter structure Param.

    Parameters
    ----------
    x               Data matrix
    param           Parameter structure, including:
                        logv    Log of noise parameters     [3x1 double]
                        mu      Signal vector               [Nx1 double]
                        A       Amplitude vector            [Mx1 double]
                        eta     Delay vector                [Mx1 double]
                        ts      Sampling time               [double]
                        D       Derivative matrix           [NxN double]
    varargin        Variables to fix for gradient calculation
                        logv    Log of noise parameters     [logical]
                        mu      Signal vector               [logical]
                        A       Amplitude vector            [logical]
                        eta     Delay vector                [logical]

    Returns
    -------
    nll             Negative log-likelihood function
    gradnll         Gradient of the negative log-likelihood function
    """

    # Parse function inputs
    [N, M] = x.shape
    # validateattributes(x, {'double'}, {'2d'})
    # validateattributes(Param, {'struct'}, {'nonempty'})
    # validateattributes(Fix, {'struct'}, {'nonempty'})

    # Parse parameter dictionary
    Pfields = param.keys()
    Ignore = dict()
    if 'logv' in Pfields:
        v = np.exp(param['logv'])
        v = np.reshape(v, (len(v), 1))
        # validateattributes(v, {'double'}, {'vector', 'numel', 3})
        # else:
        # error('TDNLL requires Param structure with logv field')
    if 'mu' in Pfields:
        mu = param['mu']
        mu = np.reshape(mu, (len(mu), 1))
        # validateattributes(mu, {'double'}, {'vector', 'numel', N})
        # else:
        # error('TDNLL requires Param structure with mu field')
    if 'a' in Pfields and param['a'] != []:
        a = param['a']
        a = np.reshape(a, (len(a), 1))
        # validateattributes(a, {'double'}, {'vector', 'numel', M})
        Ignore['a'] = False
    else:
        a = np.ones((M, 1))
        Ignore['a'] = True
    if 'eta' in Pfields and param['eta'] != []:
        eta = param['eta']
        eta = np.reshape(eta, (len(eta), 1))
        # validateattributes(eta, {'double'}, {'vector', 'numel', M})
        Ignore['eta'] = False
    else:
        eta = np.zeros((M, 1))
        Ignore['eta'] = True
    if 'ts' in Pfields:
        ts = param['ts']
        # validateattributes(ts, {'double'}, {'scalar'})
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')
    if 'd' in Pfields:
        d = param['d']
        # validateattributes(d, {'double'}, {'size', [N N]})
    else:
        # Compute derivative matrix
        fun = lambda theta, w: - 1j * w
        d = tdtf(fun, 0, N, ts)

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(N, ts)
    w = 2 * np.pi * f
    w = w.reshape(len(w), 1)
    mu_f = np.fft.fft(mu.flatten()).reshape(len(mu), 1)

    gradcalc = np.logical_not([[Fix['logv']], [Fix['mu']], [Fix['a'] or Ignore['a']], [Fix['eta'] or Ignore['eta']]])

    if Ignore['eta']:
        zeta = mu * np.conj(a).T
        zeta_f = np.fft.fft(zeta, axis=0)
    else:
        exp_iweta = np.exp(1j * np.tile(w, M) * np.conj(np.tile(eta, N)).T)
        zeta_f = np.conj(np.tile(a, N)).T * np.conj(exp_iweta) * np.tile(mu_f, M)
        zeta = np.real(np.fft.ifft(zeta_f, axis=0))

    # Compute negative - log likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Simplest case: just variance and signal parameters, A and eta fixed at
    # defaults
    if Ignore['a'] and Ignore['eta']:
        Dmu = np.real(np.fft.ifft(1j * w * mu_f, axis=0))
        valpha = v[0]
        vbeta = v[1] * mu**2
        vtau = v[2] * Dmu**2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / np.tile(vtot, M)
        nll = M * N * np.log(2 * np.pi) / 2 + (M / 2) * np.sum(np.log(vtot)) + np.sum(resnormsq) / 2

        # Compute gradient if requested
        # if nargout > 1:
        Ngrad = np.sum(gradcalc[0:2] * [[3], [N]])
        gradnll = np.zeros((Ngrad, 1))
        nStart = 0
        dvar = (vtot - np.mean(ressq, axis=1).reshape(N, 1)) / vtot**2
        if gradcalc[0]:
            gradnll[nStart] = (M / 2) * np.sum(dvar) * v[0]
            gradnll[nStart+1] = (M / 2) * np.sum(mu**2 * dvar) * v[1]
            gradnll[nStart+2] = (M / 2) * np.sum(Dmu**2. * dvar) * v[2]
            nStart = nStart + 3
        if gradcalc[1]:
            print('mu shape : ', mu.shape)
            print('dvar shape: ', dvar.shape)
            print('d shape: ', d.shape)
            print('Dmu shape: ', Dmu.shape)
            gradnll[nStart:nStart + N] = M * (v[1] * mu * dvar + v[2] * np.dot(d.T, (Dmu * dvar)) - np.mean(res, axis=1).reshape(N,1) / vtot)

    # Alternative case: A, eta, or both are not set to defaults
    else:
        Dzeta = np.real(np.fft.ifft(1j * np.tile(w, M) * zeta_f, axis=0))

        valpha = v[0]
        vbeta = v[1] * zeta**2
        vtau = v[2] * Dzeta**2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / vtot
        nll = M * N * np.log(2 * np.pi) / 2 + np.sum(np.log(vtot)) / 2 + np.sum(resnormsq) / 2

        # Compute gradient if requested
        # if nargout > 1:
        Ngrad = np.sum(gradcalc * [[3], [N], [M], [M]])
        gradnll = np.zeros((Ngrad, 1))
        nStart = 0
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot**2
        if gradcalc[0]:
            # Gradient wrt logv
            gradnll[nStart] = (1 / 2) * np.sum(dvar) * v[0]
            gradnll[nStart + 1] = (1 / 2) * np.sum(zeta.flatten()**2 * dvar.flatten())*v[1]
            gradnll[nStart + 2] = (1 / 2) * np.sum(Dzeta.flatten()**2 * dvar.flatten())*v[2]
            nStart = nStart + 3
        if gradcalc[1]:
            # Gradient wrt mu
            P = np.fft.fft(v[1] * dvar * zeta - reswt, axis=0) - 1j * v[2] * w * np.fft.fft(dvar * Dzeta, axis=0)
            gradnll[nStart:nStart + N] = np.sum(np.conj(a).T * np.real(np.fft.ifft(exp_iweta * P, axis=0)), axis=1).reshape(N, 1)
            nStart = nStart + N
        if gradcalc[2]:
            # Gradient wrt A
            term = (vtot - valpha) * dvar - reswt * zeta
            gradnll[nStart:nStart + M] = np.conj(np.sum(term, axis=0)).reshape(M, 1) / a
            if not Fix['mu']:
                gradnll = np.delete(gradnll, nStart)
                nStart = nStart + M - 1
            else:
                nStart = nStart + M
        if gradcalc[3]:
            # Gradient wrt eta
            DDzeta = np.real(np.fft.ifft(-np.tile(w, M)**2 * zeta_f, axis=0))
            gradnll[nStart:nStart + M] = -np.sum(dvar * (zeta * Dzeta * v[1] + Dzeta * DDzeta * v[2]) - reswt * Dzeta, axis=0).reshape(M,1)
            if not Fix['mu']:
                gradnll = np.delete(gradnll, nStart)
    gradnll = gradnll.flatten()

    return nll, gradnll
