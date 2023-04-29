import numpy as np
from thztoolsPY.shiftmtx import shiftmtx
import time
from thztoolsPY.airscancorrect import airscancorrect
from thztoolsPY.tdtf import tdtf
from thztoolsPY.tdnoisefit import tdnoisefit


def calcnoise(t, x):
    # determine sampling time
    dt = np.diff(t)
    ts = np.mean(dt)

    # compute data size
    n, m = x.shape

    # initialize parameter dictionary
    fix = {}
    ignore = {}
    param = {}
    output = {'p': {}}

    start_time = time.time()
    ############################################################################################
    # ##Fit for delay
    # #Assume constant noise, average signal, and constant amplitude

    print('Fit for Delay')

    fix['delay'] = {'logv': True, 'mu': True, 'a': True, 'eta': False}
    ignore['delay'] = {'a': True, 'eta': False}

    v0 = np.mean(np.var(x, 1)) * np.array([1, np.sqrt(np.finfo(float).eps), np.sqrt(np.finfo(float).eps)])

    mu0 = np.mean(x, axis=1)
    param['delay'] = {'v0': v0, 'mu0': mu0, 'ts': ts}
    eta0 = tdnoisefit(x, param['delay'], fix['delay'], ignore['delay'])[0]['eta']

    print("Elapsed time is", time.time() - start_time)

    ##########################################################################################
    # Fit for amplitude
    # Assume constant noise, average signal, and delays from previous fit

    print('Fit for Amplitude')
    fix['amp'] = {'logv': True, 'mu': True, 'a': False, 'eta': True}
    ignore['amp'] = {'a': False, 'eta': True}
    param['amp'] = {'v0': v0, 'mu0': mu0, 'eta0': eta0, 'ts': ts}
    a0 = tdnoisefit(x, param['amp'], fix['amp'], ignore['amp'])[0]['a']

    print("Elapsed time is", time.time() - start_time)

    #######################################################################################
    # Revise mu0

    print('Adjust x')
    param_xadjust = {'eta': eta0, 'a': a0, 'ts': ts}
    xadjusted = airscancorrect(x, param_xadjust)
    mu0 = np.mean(xadjusted, 1)

    ######################################################################################
    # Fit for var
    #  #Assume constant signal, amplitude, and delays from previous fits

    print('Fit for variance')
    fix['var'] = {'logv': False, 'mu': True, 'a': True, 'eta': True}
    ignore['var'] = {'a': False, 'eta': False}
    param['var'] = {'v0': v0, 'mu0': mu0, 'a0': a0, 'eta': eta0, 'ts': ts}
    v0 = tdnoisefit(x, param['var'], fix['var'], ignore['var'])[0]['var']

    print("Elapsed time is", time.time() - start_time)

    ###################################################################################
    # Fit for all parameters

    print('Fit for all parameters')
    fix['all'] = {'logv': False, 'mu': False, 'a': False, 'eta': False}
    ignore['all'] = {'a': False, 'eta': False}
    param['all'] = {'v0': v0, 'mu0': mu0, 'a': a0, 'eta': eta0, 'ts': ts}
    all = tdnoisefit(x, param['all'], fix['all'], ignore['all'])

    # Return results through Output Structure
    output['x'] = x
    output['t'] = t
    output['p']['eta'] = all[0]['eta']
    output['p']['a'] = all[0]['a']
    output['p']['var'] = all[0]['var']
    output['p']['mu'] = all[0]['mu']
    output['diagnostic'] = all[2]
    output['nllmin'] = all[1]

    # compare model to measurements
    vest = output['p']['var']
    muest = output['p']['mu']
    aest = output['p']['a']
    etaest = output['p']['eta']

    verr = all[2]['err']['var']
    verr = np.diag(verr)
    muerr = all[2]['err']['mu']
    aerr = all[2]['err']['a']
    etaerr = all[2]['err']['eta']

    veststar = vest * m / (m - 1)

    sigmaalphastar = np.sqrt(veststar[0])
    sigmabetastar = np.sqrt(veststar[1])
    sigmataustar = np.sqrt(veststar[2])



    # Compute residuals
    zeta = np.zeros((n, m))
    output['zeta'] = zeta
    s = np.zeros((n, n, m))

    for i in range(0, m):
        s[:, :, i] = shiftmtx(etaest[i], n, ts)
        zeta[:, i] = aest[i] * s[:, :, i] @ muest

    # transfer function and transfer matrix
    fun = lambda theta, w: -1j * w
    d = tdtf(fun, 0, n, ts)
    dmu = d @ all[0]['mu']

    # compute variance as a function of time
    valpha = output['p']['var'][0]
    vbeta = output['p']['var'][1] * output['p']['mu'] ** 2
    vtau = output['p']['var'][2] * dmu ** 2
    vtot = valpha + vbeta + vtau

    delta = (x - zeta) / np.tile(np.reshape(vtot, (n, 1)), m)

    valphalow = output['p']['var'][0] - verr[0]
    vbetalow = (output['p']['var'][1] - verr[1]) * output['p']['mu'] ** 2
    vtaulow = (output['p']['var'][2] - verr[2]) * dmu ** 2
    vtotlow = valphalow + vbetalow + vtaulow

    valphahigh = output['p']['var'][0] - verr[0]
    vbetahigh = (output['p']['var'][1] - verr[1]) * output['p']['mu'] ** 2
    vtauhigh = (output['p']['var'][2] - verr[2]) * dmu ** 2
    vtothigh = valphahigh + vbetahigh + vtauhigh

    # adjust with all parameters
    xadjusted = airscancorrect(x, {'a': all[0]['a'], 'eta': all[0]['eta'], 'ts': ts})

    # data ouput throuhg calc dictionary
    calc = {}
    calc['p'] = output['p']
    calc['nllmin'] = output['nllmin']
    calc['vest'] = vest
    calc['verr'] = verr
    calc['muest'] = muest
    calc['muerr'] = muerr
    calc['aest'] = aest
    calc['aerr'] = aerr
    calc['etaest'] = etaest
    calc['etaerr'] = etaerr
    calc['t'] = t
    calc['x'] = x
    calc['zeta'] = zeta
    calc['delta'] = delta
    calc['vtot'] = vtot
    calc['vtothigh'] = vtothigh
    calc['vtotlow'] = vtotlow
    calc['xadjusted'] = xadjusted
    calc['sigma_alpha'] = sigmaalphastar
    calc['sigma_beta'] = sigmabetastar
    calc['sigma_tau'] = sigmataustar
    print('=====================================================')
    print('sigma estimates:', np.round(sigmaalphastar, 4), "pA, ", np.round(sigmabetastar * 100, 3), "%, ",
          np.round(sigmataustar * 1e3, 3), "fs")

    return calc
