import numpy as np
import h5py

from thztoolsPY.epswater import epswater


def test_epswater():
    with h5py.File('epswater_test_data.mat', 'r') as file:
        set = file['Set']
        xf = file['Set']['epswater']['f'][0]
        xt = file['Set']['epswater']['T'][0]
        dff = np.array(xf)
        dft = np.array(xt)

        for i in range(0, dff.shape[0]):
            for j in range(0, dft.shape[0]):
                f = np.array(file[set['epswater']['f'][i, j]])[0, 0]
                t = np.array(file[set['epswater']['T'][i, j]])[0, 0]
                epsr = np.array(file[set['epswater']['epsR'][i, j]])
                epsi = np.array(file[set['epswater']['epsI'][i, j]])
                fpy = epswater(f, t)
                np.testing.assert_allclose(epsr, np.real(fpy))
                np.testing.assert_allclose(epsi, np.imag(fpy))
