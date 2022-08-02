import numpy as np

import time

from thztools.thztoolsPY.airscancorrect import airscancorrect
from thztools.thztoolsPY.tdnoisefit import tdnoisefit


def noisefitshow(t, x):
    print('===================================================')
    print('===================================================')

    # Preprocess data

    # Determine data matrix size
    n, m = x.shape

    # Determine sample time
    dt = np.diff(t)
    t = np.mean(dt)

    # Initialize parameter structure
    iFit = 1

    # Select waveform for residual plots
    idxShow = 1

    # Initialize dictionaries
    fix = {}
    ignore = {}
    param = {}
    p = {}

    start_time = time.time()
    #######################################################################################################
    # ##Fit for delay
    # #Assume constant noise, average signal, and constant amplitude

    print('Fit for Delay')

    fix['delay'] = {'logv': True, 'mu': True, 'a': True, 'eta': False}
    ignore['delay'] = {'a': True, 'eta': False}
    v0 = np.mean(np.var(x, 1)) * np.array([1, np.finfo(float).eps, np.finfo(float).eps])
    mu0 = np.mean(x, axis=1)
    param['delay'] = {'v0': v0, 'mu0': mu0, 'ts': t}
    p['delay'] = tdnoisefit(x, param['delay'], fix['delay'], ignore['delay'])
    eta0 = p['delay'][0]['eta']

    print("Elapsed time is", time.time() - start_time)

    ####################################################################################################
    # Fit for amplitude
    # Assume constant noise, average signal, and delays from previous fit

    print('Fit for Amplitude')
    fix['amp'] = {'logv': True, 'mu': True, 'a': False, 'eta': True}
    ignore['amp'] = {'a': False, 'eta': True}
    param['amp'] = {'v0': v0, 'mu0': mu0, 'eta0': eta0, 'ts': t}
    p['amp'] = tdnoisefit(x, param['amp'], fix['amp'], ignore['amp'])
    a0 = p['amp'][0]['a']

    print("Elapsed time is", time.time() - start_time)

    ########################################################################################################
    ## Revise mu0

    print('Adjust x')
    param_xadjust = {'eta': eta0, 'a': a0, 'ts': t}
    xadjusted = airscancorrect(x, param_xadjust)
    mu0 = np.mean(xadjusted, 1)

    # # ###################################################################################
    # # #Fit for var
    # # #Assume constant signal, amplitude, and delays from previous fits

    print('Fit for variance')
    fix['var'] = {'logv': False, 'mu': True, 'a': True, 'eta': True}
    ignore['var'] = {'a': False, 'eta': False}
    param['var'] = {'v0': v0, 'mu0': mu0, 'a0': a0, 'eta': eta0, 'ts': t}
    p['var'] = tdnoisefit(x, param['var'], fix['var'], ignore['var'])
    v0 = p['var'][0]['var']

    print("Elapsed time is", time.time() - start_time)

    # ######################################################################################################
    # # ## Fit for all parameters

    print('Fit for all parameters')
    fix['all'] = {'logv': False, 'mu': False, 'a': False, 'eta': False}
    ignore['all'] = {'a': False, 'eta': False}
    param['all'] = {'v0': v0, 'mu0': mu0, 'a': a0, 'eta': eta0, 'ts': t}
    p['all'] = tdnoisefit(x, param['all'], fix['all'], ignore['all'])

    print("Elapsed time is", time.time() - start_time)

    # # Assing variables

    # vest = out['all'][0]['var']
    # muest = out['all'][0]['mu']
    # aest = out['all'][0]['a']
    # etaest = out['delay'][0]['eta']

    # verr = out['all'][2]['err']['var']

    # veststar = vest * m / (m - 1)
    # verrstar = verr * m / (m - 1)

    # sigmaalphastar = np.sqrt(veststar[0])
    # sigmabetastar = np.sqrt(veststar[1])
    # sigmataustar = np.sqrt(veststar[2])

    # sigmaalphastarerr = 0.5 * verrstar[0] / sigmaalphastar
    # sigmabetastarerr = 0.5 * verrstar[1] / sigmabetastar
    # sigmataustaterr = 0.5 * verrstar[2] / sigmataustar

    # # Compute time-dependent noise amplitudes

    # def fun(theta, w):
    #   return -1j*w

    # # Determine sample time
    # dt = np.diff(t)
    # T = np.mean(dt)

    # d = tdtf(fun, 0, n, T)
    # dmu = d@out['all'][0]['mu']

    # # Compute variance as a function of time

    # valphastar = veststar[0]
    # vbetastar = veststar[1] * (out['all'][0]['mu'])**2
    # vtaustar = veststar[2] * dmu**2
    # vtotstar = valphastar + vbetastar + vtaustar

    # # Compute noise amplitude as a function of time
    # sigmatotstar = np.sqrt(vtotstar)

    # # Compute residuals
    # zeta = np.zeros((n, m))
    # s = np.zeros((n, n, m))

    # for i in range(0, m):
    #   s[:, :, i] = shiftmtx(etaest[i], n, T)
    #   zeta[:, i] = aest[i] * s[:, :, i] @ muest

    return p
