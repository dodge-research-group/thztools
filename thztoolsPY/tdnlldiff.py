import numpy as np

from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.tdtf import tdtf


def tdnlldiff(x, param, fix={'logv': False, 'mu': False, 'a': False, 'eta': False}):
    """TDNLLDIFF computes negative log-likelihood for the time-domain noise model

     Syntax:   nll = tdnlldiff(x,Param,Fix)

     Description:

     TDNLL computes the negative log-likelihood function for obtaining the
     data matrix x, given the parameter structure Param.

     Inputs:
        x       Data matrix
        Param   Parameter structure, including:
        .logv   Log of noise parameters	[3x1 double]
        .mu     Signal vector           [Nx1 double]
        .A      Amplitude vector        [Mx1 double]
        .eta    Delay vector            [Mx1 double]
        .ts     Sampling time           [double]
        .D      Derivative matrix       [NxN double]
        Fix     Variables to fix for gradient calculation
        .logv   Log of noise parameters [logical]
        .mu     Signal vector           [logical]
        .A      Amplitude vector        [logical]
        .eta    Delay vector            [logical]

     Outputs:
       nll     Negative log-likelihood function (float)
       gradnll Gradient of the negative log-likelihood function (arraylist)



    """

    global mu, v
    n, m = x.shape
    # Parse parameter dictionary
    pfields = param.keys()
    ignore = {}

    if 'logv' in pfields:
        v = np.exp(param['logv'])
    else:
        # error('TDNLL requires Param structure with logv field')
        pass
    if 'mu' in pfields:
        mu = (param['mu'])
    else:
        # error('TDNLL requires Param structure with mu field')
        pass
    if ('a' in pfields) and (param['a'] is not None):
        a = param['a']
        ignore['a'] = False
    else:
        a = np.ones((m, 1))
        ignore['a'] = True
    if ('eta' in pfields) and (param['eta'] is not None):
        eta = param['eta']
        ignore['eta'] = False
    else:
        eta = np.zeros((m, 1))
        ignore['eta'] = True
    if 'ts' in pfields:
        ts = param['ts']
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')
    if 'd' in pfields:
        d = param['d']
    else:
        def fun(theta, w):
            return -1j * w

        d = tdtf(fun, 0, n, ts)

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(n, ts)
    w = 2 * np.pi * f
    mu_f = np.fft.fft(mu)

    gradcalc = np.logical_not([fix.get('logv'), fix.get('mu'), fix['A'] or ignore['A'],
                               fix['eta'] or ignore['eta']])

    if ignore['eta']:
        zeta = mu * np.transpose(np.conjugate(a))
        zeta_f = np.fft.fft(zeta)  # ask about fft for a (N, M) matrix

    else:
        exp_iweta = np.exp(1j * np.tile(np.reshape(w, (len(w), 1)), m)) * np.transpose(np.conjugate(np.tile(eta, n)))
        zeta_f = np.transpose(np.tile(a, n)) * np.conj(exp_iweta) * np.tile(np.reshape(mu_f, (len(mu_f), 1)), m)
        zeta = np.real(np.fft.ifft(zeta_f))

        pass

    # Compute negative-log likelihood and gradient
    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res ** 2

    # Simplest case: just variance and signal parameters, A and eta fixed at defaults
    if ignore['a'] and ignore['eta']:
        dmu = np.fft.ifft(1j * (w * mu_f))
        valpha = v[0]
        vbeta = v[1] * (mu ** 2)
        vtau = v[2] * (dmu ** 2)
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / np.tile(vtot, (1, m))
        nll = m * n * np.log(2 * np.pi) / 2 + (m / 2) * sum(np.log(vtot)) + sum(resnormsq) / 2

        # Compute gradient if requested
        # if nargout > 1:
        ngrad = sum(gradcalc[0:2] * np.array([3, n]))
        gradnll = np.zeros((ngrad, 1))
        nstart = 0
        dvar = (vtot - np.mean(ressq, axis=1)) / (vtot ** 2)

        if gradcalc[0]:
            gradnll[nstart] = (m / 2) * sum(dvar) * v[0]
            gradnll[nstart + 1] = (m / 2) * sum(mu ** 2 * dvar) * v[1]
            gradnll[nstart + 2] = (m / 2) * sum(dmu ** 2 * dvar) * v[2]
            nstart = nstart + 3
        pass

        if gradcalc[1]:
            gradnll[nstart:(nstart + n - 1)] = m * (v[1] * mu * dvar + v[2] @ np.transpose(d) @ (dmu * dvar)
                                                    - np.mean(res, axis=1) / vtot)
        pass

    # Alternative case: A, era, or both are not set to defaults
    else:
        res = np.diff(res, 1, axis=1) / np.sqrt(2)
        ressq = res ** 2

        dzeta = np.real(np.fft.ifft(1j * (np.tile(np.reshape(w, (len(w), 1)), m) * zeta_f)))

        valpha = v[0]
        vbeta = v[2] * ((zeta[:, 1:m - 1]) ** 2 + zeta[:, 2:m] ** 2) / 2
        vtau = v[3] * ((dzeta[:, 1:m - 1]) ** 2 + dzeta[:, 2:m] ** 2) / 2
        vtot = valpha + vbeta + vtau

        resnormsq = ressq / vtot
        nll = (m - 1) * n * np.log(2 * np.pi) / 2 + sum(np.log(vtot)) / 2 + sum(resnormsq) / 2

        # compute gradient if requested
        # if nargout >1:
        ngrad = np.sum(gradcalc * [[3], [n], [m], [m]])
        gradnll = np.zeros((ngrad, 1))
        nstart = 1
        reswt = res / vtot
        dvar = (vtot - ressq) / vtot ** 2
        if gradcalc[0]:
            gradnll[nstart] = (1 / 2) * np.sum(dvar) * v[0]
            gradnll[nstart + 1] = (1 / 2) * np.sum(
                np.reshape(zeta, (len(zeta), 1)) ** 2 * np.reshape(dvar, (len(dvar), 1))) * v[1]
            gradnll[nstart + 2] = (1 / 2) * np.sum(
                np.reshape(zeta, (len(zeta), 1)) ** 2 * np.reshape(dvar, (len(dvar), 1))) * v[2]
            nstart = nstart + 3
        pass

        if gradcalc[1]:
            p = np.fft.fft(v[1] * dvar * zeta - reswt) - 1j * v[2] * w * np.fft.fft(dvar * dzeta)
            gradnll[nstart: nstart + n - 1] = np.sum(np.transpose(np.conjugate(a)) *
                                                     np.real(np.fft.ifft(exp_iweta * p)), axis=2)
            nstart = nstart + 3

        pass

        if gradcalc[2]:
            # Gradient wrt A
            term = ((vtot - valpha) * dvar - reswt * zeta)
            gradnll[nstart: (nstart + m - 1)] = np.transpose(np.conj(np.sum(term, axis=0))) / a
            nstart = nstart + m

        pass

        if gradcalc[3]:
            # Gradient wrt eta
            ddzeta = np.real(np.fft.ifft(np.tile(w, (1, m))) * zeta_f)
            gradnll[nstart: (nstart + m - 1)] = -np.sum(dvar * (zeta * dzeta * v[1] + dzeta * ddzeta * v[2])
                                                        - reswt * ddzeta)
        pass
    pass
