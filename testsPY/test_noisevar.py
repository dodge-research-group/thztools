import numpy as np
import h5py

from thztools.thztoolsPY.noisevar import noisevar


def test_noisevar():
    with h5py.File('noisevar_test_data.mat', 'r') as file:
        set = file['Set']
        sigma = file['Set']['noisevar']['sigma'][0]
        mu = file['Set']['noisevar']['mu'][0]
        t = file['Set']['noisevar']['T'][0]
        vmu = file['Set']['noisevar']['Vmu'][0]
        nsigma = np.array(sigma)
        nmu = np.array(mu)
        nt = np.array(t)

        for i in range(0, nsigma.shape[0]):
            for j in range(0, nmu.shape[0]):
                for k in range(0, nt.shape[0]):
                    sigma = np.array(file[set['noisevar']['sigma'][i, j, k]])[0]
                    mu = np.array(file[set['noisevar']['mu'][i, j, k]])[0]
                    t = np.array(file[set['noisevar']['T'][i, j, k]])[0, 0]
                    vmu = np.array(file[set['noisevar']['Vmu'][i, j, k]])[0]
                    vmupy = noisevar(sigma, mu, t)
                    np.testing.assert_allclose(vmu, vmupy)
