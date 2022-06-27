import numpy as np
import h5py

from thztoolsPY.pulsegen import pulsegen

with h5py.File('pulsegen_test_data.mat', 'r') as file:
    Set = file['Set']
    xN = file['Set']['pulsegen']['N'][0]
    xT0 = file['Set']['pulsegen']['t0'][0]
    xw = file['Set']['pulsegen']['w'][0]
    xA = file['Set']['pulsegen']['A'][0]
    xT = file['Set']['pulsegen']['T'][0]
    dfN = np.array(xN)
    dfT0 = np.array(xT0)
    dfw = np.array(xw)
    dfA = np.array(xA)
    dfT = np.array(xT)

    for i in range(0, dfN.shape[0]):
        for j in range(0, dfT0.shape[0]):
            for k in range(0, dfw.shape[0]):
                for r in range(0, dfA.shape[0]):
                    for s in range(0, dfT.shape[0]):
                        N = np.array(file[Set['pulsegen']['N'][i, j, k, r, s]])[0, 0]
                        t0 = np.array(file[Set['pulsegen']['t0'][i, j, k, r, s]])[0, 0]
                        w = np.array(file[Set['pulsegen']['w'][i, j, k, r, s]])[0, 0]
                        A = np.array(file[Set['pulsegen']['A'][i, j, k, r, s]])[0, 0]
                        T = np.array(file[Set['pulsegen']['T'][i, j, k, r, s]])[0, 0]
                        y = np.array(file[Set['pulsegen']['y'][i, j, k, r, s]])[0]
                        fpy = pulsegen(N.astype(int), t0, w, A, T)[0]
                        np.testing.assert_allclose(y, fpy)
