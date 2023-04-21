import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from thztoolsPY.tdnoisefit import tdnoisefit


def test():
    fname = os.path.join(os.path.dirname(__file__), 'tdnoisefit_test_data.mat')
    with h5py.File(fname, 'r') as file:
        Set = file['Set']
        x = file['Set']['tdnoisefit']['x'][0]
        param = file['Set']['tdnoisefit']['paramForPy']
        fix = file['Set']['tdnoisefit']['fixForPy']
        ignore = file['Set']['tdnoisefit']['ignoreForPy']
        pMatlab = file['Set']['tdnoisefit']['P']
        diagnostic = file['Set']['tdnoisefit']['Diagnostic']
        dfx = np.array(x)
        dfParam = np.array(param)

        #for i in range(0, dfx.shape[0]):
            #for j in range(0, dfParam.shape[0]):
        for i in range(0, 1):
            for j in range(0, 1):
                x = np.array(file[Set['tdnoisefit']['x'][j, i]]).T
                param = {'v0': np.array(file[param[j, 0]]['v0'])[0],
                         'mu0': np.array(file[param[j, 0]]['mu0'])[0],
                         'a0': np.array(file[param[j, 0]]['A0'])[0],
                         'eta0': np.array(file[param[j, 0]]['eta0'])[0],
                         'ts': np.array(file[param[j, 0]]['ts'])[0]}
                fix = {'logv': bool(np.array(file[fix[j, 0]]['logv'])),
                       'mu': bool(np.array(file[fix[j, 0]]['mu'])),
                       'a': bool(np.array(file[fix[j, 0]]['A'])),
                       'eta': bool(np.array(file[fix[j, 0]]['eta']))}
                ignore = {'a': bool(np.array(file[ignore[j, 0]]['A'])),
                          'eta': bool(np.array(file[ignore[j, 0]]['eta']))}
                p = {'var': np.array(file[pMatlab[j, 0]]['var'])[0],
                     'mu': np.array(file[pMatlab[j, 0]]['mu'])[0],
                     'a': np.array(file[pMatlab[j, 0]]['A'])[0],
                     'eta': np.array(file[pMatlab[j, 0]]['eta'])[0],
                     'ts': np.array(file[pMatlab[j, 0]]['ts'])[0]}
                fun = np.array(file[Set['tdnoisefit']['fval'][j, i]])[0]
                diagnostic = np.array(file[Set['tdnoisefit']['Diagnostic'][j, i]])[0]
                [pPy, funPy, diagnosticPy] = tdnoisefit(x, param, fix, ignore)
                print('Matlab Costfun: ' + str(fun))
                print('Python Costfun: ' + str(funPy))
                print('-------------------')
                print('Matlab var: ' + str(p['var']))
                print('Python var: ' + str(pPy['var']))
                # np.testing.assert_allclose(funPy, fun, atol=1e-02)
                # np.testing.assert_allclose(pPy['var'], p['var'], atol=1e-02)
                # np.testing.assert_allclose(pPy['mu'], p['mu'], atol=1e-02)
                # np.testing.assert_allclose(pPy['a'], p['a'], atol=1e-02)
                # np.testing.assert_allclose(pPy['eta'], p['eta'], atol=1e-02)
                # np.testing.assert_allclose(pPy['ts'], p['ts'], atol=1e-02)
                # np.testing.assert_allclose(diagnosticPy, diagnostic)
                fig = plt.figure()
                plt.title(r'$\mu$')
                plt.plot(x[:, 0], label='Input + Noise')
                plt.plot(pPy['mu'], label = 'Fit Py')
                plt.plot(p['mu'], label='Fit Mat', linestyle='dashed')
                plt.legend()
                plt.show()
                fig = plt.figure()
                plt.title('A')
                plt.plot(pPy['a'], label='Fit Py')
                plt.plot(p['a'], label='Fit Mat', linestyle='dashed')
                plt.legend()
                plt.show()
                fig = plt.figure()
                plt.title(r'$\eta$')
                plt.plot(pPy['eta'], label='Fit Py')
                plt.plot(p['eta'], label='Fit Mat', linestyle='dashed')
                plt.legend()
                plt.show()
                fig = plt.figure()
                plt.title(r'var')
                plt.plot(pPy['var'], label='Fit Py')
                plt.plot(p['var'], label='Fit Mat', linestyle='dashed')
                plt.legend()
                plt.show()
