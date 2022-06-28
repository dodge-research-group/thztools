import numpy as np
import h5py

from thztools.thztoolsPY.noisevar import noisevar

with h5py.File('noisevar_test_data.mat', 'r') as file:
    Set = file['Set']
    sigma = file['Set']['noisevar']['sigma']
    mu = file['Set']['noisevar']['mu']
    T = file['Set']['noisevar']['T']
    Vmu = file['Set']['noisevar']['Vmu']
    nsigma = np.array(sigma)
    nmu = np.array(mu)
    nT = np.array(T)

    for i in range(0, nsigma.shape[0]):
        for j in range(0, nmu.shape[0]):
            sigma = np.concatenate(np.array(file[Set['noisevar']['sigma'][i, j]]))
            mu = np.concatenate(np.array(file[Set['noisevar']['mu'][i, j]]))
            T = np.array(file[Set['noisevar']['T'][i, j]])[0][0]
            Vmu = (np.array(file[Set['noisevar']['Vmu'][i, j]]))[0]
            Vmupy = noisevar(sigma, mu, T)
            np.testing.assert_allclose(Vmu, Vmupy)
