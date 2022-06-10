import numpy as np
import h5py

from thztoolsPY.epswater import epswater

with h5py.File('epswater_test_data.mat', 'r') as file:
    Set = file['Set']
    xF = file['Set']['epswater']['F'][0]
    xT = file['Set']['epswater']['T'][0]
    dfF = np.array(xF)
    dfT = np.array(xT)

    for i in range(0, dfF.shape[0]):
        for j in range(0, dfT.shape[0]):
            f = np.array(file[Set['epswater']['F'][i, j]])
            t = np.array(file[Set['epswater']['T'][i, j]])
            epsilonr = np.array(file[Set['epswater']['epsilonr'][i, j]])
            fpy = epswater(f, t)
            np.testing.assert_allclose(epsilonr, fpy)
