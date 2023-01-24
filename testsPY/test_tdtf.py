import numpy as np
import h5py

from thztoolsPY.tdtf import tdtf


def test_tdtf():
    def fun(theta, w):
        return theta[0] * np.exp(-1j * theta[1] * w)

    with h5py.File('tdtf_test_data.mat', 'r') as file:
        set = file['Set']
        theta = file['Set']['tdtf']['theta'][0]
        n = file['Set']['tdtf']['N'][0]
        ts = file['Set']['tdtf']['ts'][0]

        ntheta = np.array(theta)
        nn = np.array(n)
        nts = np.array(ts)

        for i in range(0, ntheta.shape[0]):
            for j in range(0, nn.shape[0]):
                for k in range(0, nts.shape[0]):
                    theta = np.array(file[set['tdtf']['theta'][i, j, k]])[0]
                    n = np.array(file[set['tdtf']['N'][i, j, k]])[0, 0]
                    ts = np.array(file[set['tdtf']['ts'][i, j, k]])[0, 0]
                    h = np.array(file[set['tdtf']['h'][i, j, k]])
                    hpy = tdtf(fun, theta, n, np.array([ts]))
                    h = np.transpose(h)
                    np.testing.assert_allclose(hpy, h, atol=1e-10)


