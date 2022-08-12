import numpy as np
import time

from matplotlib import pyplot as plt

from thztoolsPY.airscancorrect import airscancorrect
from thztoolsPY.shiftmtx import shiftmtx
from thztoolsPY.tdnoisefit import tdnoisefit
from thztoolsPY.tdtf import tdtf


def noisefitshow(t, x):
    print('===================================================')
    print('===================================================')

    # Preprocess data

    # Determine data matrix size
    n, m = x.shape

    # Determine sampling time
    dt = np.diff(t)
    ts = np.mean(dt)

    # Initialize parameter structure
    ifit = 1

    # Select waveform for residual plots
    idxshow = 1

    # Initialize dictionaries
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
    # #
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

    print("Elapsed time is", time.time() - start_time)

    ###################################################################################
    # Assign variables

    vest = output['p']['var']
    muest = output['p']['mu']
    aest = output['p']['a']
    etaest = output['p']['eta']
    verr = all[2]['err']['var']

#
    veststar = vest * m / (m - 1)
    verrstar = verr * m / (m - 1)

    sigmaalphastar = np.sqrt(veststar[0])
    sigmabetastar = np.sqrt(veststar[1])
    sigmataustar = np.sqrt(veststar[2])

    sigmaalphastarerr = 0.5 * verrstar[0] / sigmaalphastar
    sigmabetastarerr = 0.5 * verrstar[1] / sigmabetastar
    sigmataustaterr = 0.5 * verrstar[2] / sigmataustar

    # Compute time-dependent noise amplitudes

    def fun(theta, w):
        return -1j * w

    # Transfer matrix
    d = tdtf(fun, 0, n, ts)
    dmu = d @ all[0]['mu']

    # Compute variance as a function of time

    valphastar = veststar[0]
    vbetastar = veststar[1] * (all[0]['mu']) ** 2
    vtaustar = veststar[2] * dmu ** 2
    vtotstar = valphastar + vbetastar + vtaustar

    # Compute noise amplitude as a function of time
    sigmatotstar = np.sqrt(vtotstar)
    output['sigmatotstar'] = sigmatotstar

    # Compute residuals
    zeta = np.zeros((n, m))
    output['zeta'] = zeta
    s = np.zeros((n, n, m))

    for i in range(0, m):
        s[:, :, i] = shiftmtx(etaest[i], n, ts)
        zeta[:, i] = aest[i] * s[:, :, i] @ muest

    delta = (x - zeta) / np.tile(np.reshape(vtotstar, (1001, 1)), 50)
    output['delta'] = delta

    # Compute variance of adjusted waveform data
    xadjusted = airscancorrect(x, {'a': all[0]['a'], 'eta': all[0]['eta']})
    output['xadjusted'] = xadjusted

    plt.figure(figsize=(10, 8))
    plt.title('Noise Model Results, 50 ps ', fontsize=15)
    plt.plot(t, sigmatotstar, color='red', linewidth=2)
    plt.xlabel('Time (ps)', fontsize=15)
    plt.ylabel('$\sigma \hat{} ^{*}$ (nA)', fontsize=15)
    #plt.legend(fontsize=15)
    plt.show()

    ##########
    plt.figure(figsize=(10, 8))
    plt.title('Drift Amplitude, 50 ps', fontsize=15)
    plt.stem(np.arange(0, 50), 1e2 * (aest / aest[0] - 1), linefmt='C0')
    plt.xlabel('Iteration Number', fontsize=15)
    plt.ylabel('Drift Amplitude (% from 1)', fontsize=15)
    plt.show()

    #####
    plt.figure(figsize=(10, 8))
    plt.title('Drift Delay, 50 ps ', fontsize=15)
    plt.stem(np.arange(0, 50), 1e3 * (etaest - etaest[0]))
    plt.xlabel('Iteration Number', fontsize=15)
    plt.ylabel('Drift Delay (fs)', fontsize=15)
    plt.show()

    return output
