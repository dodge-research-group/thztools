import numpy as np
import h5py

from thztoolsPY.epswater import epswater

with h5py.File('epswater_test_data.mat', 'r') as file:
    Set = file['Set']
    xf = file['Set']['epswater']['f'][0]
    xT = file['Set']['epswater']['T'][0]
    dff = np.array(xf)
    dfT = np.array(xT)

    for i in range(0, dff.shape[0]):
        for j in range(0, dfT.shape[0]):
            f = np.array(file[Set['epswater']['f'][i, j]])[0, 0]
            t = np.array(file[Set['epswater']['T'][i, j]])[0, 0]
            epsR = np.array(file[Set['epswater']['epsR'][i, j]])
            epsI = np.array(file[Set['epswater']['epsI'][i, j]])
            fpy = epswater(f, t)
            np.testing.assert_allclose(epsR, np.real(fpy))
            np.testing.assert_allclose(epsI, np.imag(fpy))
