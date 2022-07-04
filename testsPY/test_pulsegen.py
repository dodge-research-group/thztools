import numpy as np
import h5py

from thztoolsPY.pulsegen import pulsegen


def test_pulsegen():
    with h5py.File('pulsegen_test_data.mat', 'r') as file:
        set = file['Set']
        xn = file['Set']['pulsegen']['N'][0]
        xt0 = file['Set']['pulsegen']['t0'][0]
        xw = file['Set']['pulsegen']['w'][0]
        xa = file['Set']['pulsegen']['A'][0]
        xt = file['Set']['pulsegen']['T'][0]
        dfn = np.array(xn)
        dft0 = np.array(xt0)
        dfw = np.array(xw)
        dfa = np.array(xa)
        dft = np.array(xt)

        for i in range(0, dfn.shape[0]):
            for j in range(0, dft0.shape[0]):
                for k in range(0, dfw.shape[0]):
                    for r in range(0, dfa.shape[0]):
                        for s in range(0, dft.shape[0]):
                            n = np.array(file[set['pulsegen']['N'][i, j, k, r, s]])[0, 0]
                            t0 = np.array(file[set['pulsegen']['t0'][i, j, k, r, s]])[0, 0]
                            w = np.array(file[set['pulsegen']['w'][i, j, k, r, s]])[0, 0]
                            a = np.array(file[set['pulsegen']['A'][i, j, k, r, s]])[0, 0]
                            t = np.array(file[set['pulsegen']['T'][i, j, k, r, s]])[0, 0]
                            y = np.array(file[set['pulsegen']['y'][i, j, k, r, s]])[0]
                            fpy = pulsegen(n.astype(int), t0, w, a, t)[0]
                            np.testing.assert_allclose(y, fpy)
