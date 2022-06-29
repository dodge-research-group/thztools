import h5py
import numpy as np

from thztoolsPY.fftfreq import fftfreq

def test():
    with h5py.File('fftfreq_test_data.mat', 'r') as file:
        Set = file['Set']
        xN = file['Set']['fftfreq']['N'][0]
        xT = file['Set']['fftfreq']['T'][0]
        dfN = np.array(xN)
        dfT = np.array(xT)

        for i in range(0, dfN.shape[0]):
            for j in range(0, dfT.shape[0]):
                N = np.array(file[Set['fftfreq']['N'][i, j]])[0, 0]
                T = np.array(file[Set['fftfreq']['T'][i, j]])[0, 0]
                y = np.array(file[Set['fftfreq']['f'][i, j]])[0]
                fpy = fftfreq(N.astype(int), T)
                np.testing.assert_allclose(y, fpy)
