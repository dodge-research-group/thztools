import h5py
import numpy as np

from thztoolsPY.thzgen import thzgen


def test_thzgen():
    with h5py.File('thzgen_test_data.mat', 'r') as file:
        set = file['Set']
        xn = file['Set']['thzgen']['N'][0]
        xt = file['Set']['thzgen']['T'][0]
        xt0 = file['Set']['thzgen']['t0'][0]
        dfn = np.array(xn)
        dft = np.array(xt)
        dft0 = np.array(xt0)

        for i in range(0, dfn.shape[0]):
            for j in range(0, dft.shape[0]):
                for k in range(0, dft0.shape[0]):
                    n = np.array(file[set['thzgen']['N'][i, j, k]])[0, 0]
                    t = np.array(file[set['thzgen']['T'][i, j, k]])[0, 0]
                    t0 = np.array(file[set['thzgen']['t0'][i, j, k]])[0, 0]
                    y = np.array(file[set['thzgen']['y'][i, j, k]])[0]
                    fpy = thzgen(n.astype(int), t, t0, varargin=1)[0]
                    np.testing.assert_allclose(y, fpy)
