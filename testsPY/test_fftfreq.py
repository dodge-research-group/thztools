import h5py
import numpy as np

from thztools.thztoolsPY.fftfreq import fftfreq


def test_fftfreq():
    with h5py.File('fftfreq_test_data.mat', 'r') as file:
        set = file['Set']
        xn = file['Set']['fftfreq']['N'][0]
        xt = file['Set']['fftfreq']['T'][0]
        dfn = np.array(xn)
        dft = np.array(xt)

        for i in range(0, dfn.shape[0]):
            for j in range(0, dft.shape[0]):
                n = np.array(file[set['fftfreq']['N'][i, j]])[0, 0]
                t = np.array(file[set['fftfreq']['T'][i, j]])[0, 0]
                y = np.array(file[set['fftfreq']['f'][i, j]])[0]
                fpy = fftfreq(n.astype(int), t)
                np.testing.assert_allclose(y, fpy)
