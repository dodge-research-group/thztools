import os

import h5py
import numpy as np

from thztoolsPY.fftfreq import fftfreq


def test_fftfreq():
    fname = os.path.join(os.path.dirname(__file__), 'costfunlsq_test_data.mat')

    with h5py.File(fname, 'r') as file:
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


def test():
    fname = os.path.join(os.path.dirname(__file__), 'costfunlsq_test_data.mat')
    with h5py.File(fname, 'r') as file:
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
