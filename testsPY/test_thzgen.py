import h5py
import numpy as np

from thztools.thztoolsPY.thzgen import thzgen

with h5py.File('thzgen_test_data.mat', 'r') as file:
    Set = file['Set']
    xN = file['Set']['thzgen']['N'][0]
    xT = file['Set']['thzgen']['T'][0]
    xT0 = file['Set']['thzgen']['t0'][0]
    dfN = np.array(xN)
    dfT = np.array(xT)
    dfT0 = np.array(xT0)

    for i in range(0, dfN.shape[0]):
        for j in range(0, dfT.shape[0]):
            for k in range(0, dfT0.shape[0]):
                N = np.array(file[Set['thzgen']['N'][i, j, k]])[0, 0]
                T = np.array(file[Set['thzgen']['T'][i, j, k]])[0, 0]
                t0 = np.array(file[Set['thzgen']['t0'][i, j, k]])[0, 0]
                y = np.array(file[Set['thzgen']['y'][i, j, k]])[0]
                fpy = thzgen(N.astype(int), T, t0, varargin=1)[0]
                np.testing.assert_allclose(y, fpy)
