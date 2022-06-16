import numpy as np

from thztools.thztoolsPY.tdtf import tdtf

from thztools.thztoolsPY.fftfreq import fftfreq

def tdnlldiff(x, param, *args):

    # Parse function inputs
    n, m = x.shape
    if len(args) > 2:
        fix = args[0]
    else:
        fix = {'logv':False, 'mu':False, 'A':False, 'eta':False}

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
    if ('A' in pfields) and (param['A'] != None):
        A = param['A']
        ignore['A'] = False
    else:
        A = np.ones((m, 1))
        ignore['A'] = True
    if ('eta' in pfields) and (param['eta'] != None):
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
    if 'D' in pfields:
        d = param['D']
    else:
        fun = lambda theta, w: -1j*w
        d = tdtf (fun, 0, n, ts)

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(n, ts)
    w = 2*np.pi*f
    mu_f = np.fft.fft(mu)

    gradcalc = np.logical_not([fix.get('logv'), fix.get('mu'), fix['A'] or ignore['A'],
                               fix['eta'] or ignore['eta']])

    if ignore['eta']:
        zeta = mu*A.H
        zeta_f = np.fft.fft(zeta) # ask about fft for a (N, M) matrix
    else:
        # w1 = np.reshape(w, (len(w), 1))  # reshape the row vector w (1xn) into a column vector (nx1)
        # w2 = np.tile(w1,m)               # "expand" the nx1 vector into a nxm matrix with each column equal to the vector w1

        # eta1 = np.tile(eta, n)           # "expand" the mx1 vector into a mxn matrix with each column equal to the vector eta


        # exp_iweta = np.exp(1j*w2)*eta1
        exp_iweta = np.exp(1j*np.tile(np.reshape(w, (len(w), 1)),  m)) @ np.tile(eta, n)      # do the same in only one line
        zeta_f = np.tile(A, n) @ np.conj(exp_iweta) @ np.tile(np.reshape(mu_f, (len(mu_f), 1)), m)
        zeta = np.real(np.fft.ifft(zeta_f))
        pass

    # Compute negative-log likelihood and gradient

    # Compute ressiduals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Simplest case: just variance and signal parameters, A and eta fixed at defaults
    if ignore['A'] and ignore['eta']:
        dmu = np.fft.ifft(1j*(w*mu_f))
        valpha = v[0]
        vbeta = v[1]*(mu**2)
        vtau = v[2]*(dmu**2)
        vtot = valpha + vbeta + vtau

        resnormsq = ressq/np.tile(vtot, (1, m))
        nll = m*n*np.log(2*np.pi)/2 + (m/2)*sum(np.log(vtot)) + sum(resnormsq(:))/2

    # Compute graient if requested
        #if nargout > 1:                # we should ignore this line by now
        ngrad = sum(gradcalc[0:2] @ np.reshape(np.array([3, n])), (2, 1))
        gradnll = np.zeros((ngrad[0], 1))
        nstart = 0
        dvar = (vtot - np.mean(ressq, axis=1))/ (vtot**2)

        if gradcalc[0]:
            gradnll[nstart] = (m/2)*sum(dvar)*v[0]
            gradnll[nstart+1] = (m/2)*sum(mu**2 * dvar)*v[1]
            gradnll[nstart+2] = (m/2)*sum(dmu**2 * dvar)*v[2]
            nstart = nstart + 3
        pass

        if gradcalc[1]:
            for i in range(n):
                gradnll[nstart+1] = m*(v[2]*mu*dvar + v[2]*np.transpose(d)*(dmu*dvar) - np.mean(res, axis=1)/vtot)
        pass
    pass

        else:

        res = diff(res, 1, axis=1)/np.sqrt(2)
        ressq = res**2

        dzeta = np.real(np.fft.ifft(1j@(np.tile(np.reshape(w, (len(w), 1)),  m)*zeta_f)))

        valpha = v[0]

        vbeta = v[2]*((zeta[:, 1:m-1])**2 + zeta[:, 2:m]**2)/2
        vtau  = v[3]*((dzeta[:, 1:m-1])**2 + dzeta[:, 2:m]**2)/2
        vtot = valpha + vbeta + vtau

        pass
