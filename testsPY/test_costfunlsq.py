import os
import pathlib

import numpy as np
import h5py

from thztoolsPY.costfunlsq import costfunlsq


def test():
    cur_path = pathlib.Path(__file__).parent.resolve()
    fname = cur_path / 'costfunlsq_test_data.mat'
    with h5py.File(fname, 'r') as file:
        Set = file['Set']
        xtheta = file['Set']['costfunlsq']['theta'][0]
        x_xx = file['Set']['costfunlsq']['xx'][0]
        x_yy = file['Set']['costfunlsq']['yy'][0]
        xsigmax = file['Set']['costfunlsq']['sigmax'][0]
        xsigmay = file['Set']['costfunlsq']['sigmay'][0]
        xwfft = file['Set']['costfunlsq']['wfft'][0]
        dftheta = np.array(xtheta)
        dfxx = np.array(x_xx)
        dfyy = np.array(x_yy)
        dfsigmax = np.array(xsigmax)
        dfsigmay = np.array(xsigmay)
        dfwfft = np.array(xwfft)

        fun = lambda theta, wfft: theta[0] * np.exp(1j * theta[1] * wfft)

        for i in range(0, dftheta.shape[0]):
            for j in range(0, dfxx.shape[0]):
                for k in range(0, dfyy.shape[0]):
                    for l in range(0, dfsigmax.shape[0]):
                        for m in range(0, dfsigmay.shape[0]):
                            for n in range(0, dfwfft.shape[0]):
                                theta = np.array(file[Set['costfunlsq']['theta'][i, j, k, l, m, n]])[0]
                                xx = np.array(file[Set['costfunlsq']['xx'][i, j, k, l, m, n]])[0]
                                yy = np.array(file[Set['costfunlsq']['yy'][i, j, k, l, m, n]])[0]
                                sigmax = np.array(file[Set['costfunlsq']['sigmax'][i, j, k, l, m, n]])[0]
                                sigmay = np.array(file[Set['costfunlsq']['sigmay'][i, j, k, l, m, n]])[0]
                                wfft = np.array(file[Set['costfunlsq']['wfft'][i, j, k, l, m, n]])[0]
                                res = np.array(file[Set['costfunlsq']['res'][i, j, k, l, m, n]])[0]
                                fpy = costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft)
                                np.testing.assert_allclose(res, fpy)