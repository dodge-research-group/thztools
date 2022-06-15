import numpy as np

from thztools.thztoolsPY.tdtf import tdtf

from thztools.thztoolsPY.fftfreq import fftfreq

def tdnlldiff(x, param, *args):

    # Parse function inputs
    N, M = x.shape
    if len(args) > 2:
        Fix = args[0]
    else:
        Fix = {'logv':False, 'mu':False, 'A':False, 'eta':False}

    # Parse parameter dictionary
    Pfields = param.keys()
    Ignore = {}
    if 'logv' in Pfields:
        v = np.exp(param['logv'])
    else:
        # error('TDNLL requires Param structure with logv field')
        pass
    if 'mu' in Pfields:
        mu = (param['mu'])
    else:
        # error('TDNLL requires Param structure with mu field')
        pass
    if ('A' in Pfields) and (param['A'] != None):
        A = param['A']
        Ignore['A'] = False
    else:
        A = np.ones((M,1))
        Ignore['A'] = True
    if ('eta' in Pfields) and (param['eta'] != None):
        eta = param['eta']
        Ignore['eta'] = False
    else:
        eta = np.zeros((M,1))
        Ignore['eta'] = True
    if 'ts' in Pfields:
        ts = param['ts']
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')
    if 'D' in Pfields:
        D = param['D']
    else:
        fun = lambda theta, w: -1j*w
        D = tdtf (fun, 0, N, ts)

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(N, ts)
    w = 2*np.pi*f
    mu_f = np.fft.fft(mu)

    gradcalc = np.logical_not([Fix.get('logv'), Fix.get('mu'), Fix['A'] or Ignore['A'],\
                               Fix['eta'] or Ignore['eta']])

    if Ignore['eta']:
        zeta = mu*A.H
        zeta_f = np.fft.fft(zeta) # ask about fft for a (N, M) matrix
    else:
        # this else block needs completion
        # exp_iweta = np.exp(1j*w[])
        pass

    # Compute negative-log likelihood and gradient

    # Compute ressiduals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    # Simplest case: just variance and signal parameters, A and eta fixed at defaults
    if Ignore['A'] and Ignore['eta']:
        Dmu = np.fft.ifft(1j*(w*mu_f))
        valpha = v(0)
        vbeta = v(1) * (mu**2)
        vtau = v(2) * (Dmu**2)
        vtot = valpha + vbeta + vtau

        resnormsq = ressq


    pass
