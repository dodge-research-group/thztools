import numpy as np
import h5py

from thztools.thztoolsPY.tdnll import tdnll

def test():
    with h5py.File('tdnll_test_data.mat', 'r') as file:
        Set = file['Set']
        x = file['Set']['tdnll']['x'][0]
        Param = file['Set']['tdnll']['Param']
        Varargin = file['Set']['tdnll']['varargin']
        dfx = np.array(x)
        dfParam = np.array(Param)
        dfvarargin = np.array(Varargin)

        for i in range(0, dfx.shape[1]):
            for j in range(0, dfParam.shape[0]):
                for k in range(0, dfvarargin.shape[0]):
                    x = np.array(file[Set['tdnll']['x'][k, j, i]]).T
                    param = {'logv': np.array(file[Param[0, j, 0]]['logv'])[0],
                             'mu': np.array(file[Param[0, j, 0]]['mu'])[0],
                             'A': np.array(file[Param[0, j, 0]]['A'])[0],
                             'eta': np.array(file[Param[0, j, 0]]['eta'])[0],
                             'ts': np.array(file[Param[0, j, 0]]['ts'])[0],
                             'D': np.array(file[Param[0, j, 0]]['D'])}
                    varargin = {'logv': bool(np.array(file[Varargin[k, 0, 0]]['logv'])),
                                'mu': bool(np.array(file[Varargin[k, 0, 0]]['mu'])),
                                'A': bool(np.array(file[Varargin[k, 0, 0]]['A'])),
                                'eta': bool(np.array(file[Varargin[k, 0, 0]]['eta']))}
                    nll = np.array(file[Set['tdnll']['nll'][k, j, i]])[0, 0]
                    gradnll = np.array(file[Set['tdnll']['gradnll'][k, j, i]])[0]
                    [nllPy, gradnllPy] = tdnll(x, param, varargin)
                    np.testing.assert_allclose(nllPy, nll)
                    np.testing.assert_allclose(gradnllPy, gradnll)
