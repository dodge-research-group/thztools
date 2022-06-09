import numpy as np
import h5py

from thztools.thztoolsPY.thzgen import thzgen

with h5py.File('thzgen_test_data.mat', 'r') as file:
    Set = file['Set']
    for i in range(2):
        for j in range(2):
            for k in range(2):
                N =  np.array(file[Set['thzgen']['N'][i,j,k]])[0, 0]
                T =  np.array(file[Set['thzgen']['T'][i,j,k]])[0, 0]
                t0 =  np.array(file[Set['thzgen']['t0'][i,j,k]])[0, 0]
                y =  np.array(file[Set['thzgen']['y'][i,j,k]])[0]
                fpy = thzgen(N.astype(int), T, t0, varargin=1)[0]
                np.testing.assert_allclose(y, fpy)