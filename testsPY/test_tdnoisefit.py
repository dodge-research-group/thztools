import numpy as np
import h5py

from thztoolsPY.tdnoisefit import tdnoisefit


def test():
    with h5py.File('tdnoisefit_test_data.mat', 'r') as file:
        Set = file['Set']
        x = file['Set']['tdnoisefit']['x'][0]
        param = file['Set']['tdnoisefit']['paramForPy']
        fix = file['Set']['tdnoisefit']['fixForPy']
        ignore = file['Set']['tdnoisefit']['ignoreForPy']
        dfx = np.array(x)
        dfParam = np.array(param)

        for i in range(0, dfx.shape[0]):
            for j in range(0, dfParam.shape[0]):
                x = np.array(file[Set['tdnoisefit']['x'][j, i]]).T
                param = {'v0': np.array(file[param[j, 0]]['v0'])[0],
                         'mu0': np.array(file[param[j, 0]]['mu0'])[0],
                         'A0': np.array(file[param[j, 0]]['A0'])[0],
                         'eta0': np.array(file[param[j, 0]]['eta0'])[0],
                         'ts': np.array(file[param[j, 0]]['ts'])[0]}
                varargin = {'logv': bool(np.array(file[fix[j, 0]]['logv'])),
                            'mu': bool(np.array(file[fix[j, 0]]['mu'])),
                            'A': bool(np.array(file[fix[j, 0]]['A'])),
                            'eta': bool(np.array(file[fix[j, 0]]['eta']))}
                ignore = {'a': bool(np.array(file[ignore[j, 0]]['A'])),
                          'eta': bool(np.array(file[ignore[j, 0]]['eta']))}
                p = np.array(file[Set['tdnoisefit']['P'][j, i]])[0]
                fun = np.array(file[Set['tdnoisefit']['fval'][j, i]])[0]
                diagnostic = np.array(file[Set['tdnoisefit']['Diagnostic'][j, i]])[0]
                [pPy, funPy, diagnosticPy] = tdnoisefit(x, param, fix, ignore)
                np.testing.assert_allclose(pPy, p)
                np.testing.assert_allclose(funPy, fun)
                np.testing.assert_allclose(diagnosticPy, diagnostic)
