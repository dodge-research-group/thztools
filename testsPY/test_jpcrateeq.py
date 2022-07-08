import numpy as np
import h5py

from thztools.thztoolsPY.jpcrateeq import jpcrateeq

def test_jpcrateeq():
    with h5py.File('jpcrateeq_test_data2.mat', 'r') as file:
        set = file['Set']
        xt = set['jpcrateeq']['t'][0]
        xtheta = set['jpcrateeq']['theta'][0]
        dft = np.array(xt)
        dftheta = np.array(xtheta)

        for i in range(0, dft.shape[0]):
            for j in range(0, dftheta.shape[0]):
                t = np.array(file[set['jpcrateeq']['t'][i, j]])[0]
                theta = np.array(file[set['jpcrateeq']['theta'][i, j]])#[0]
                dndt = np.array(file[set['jpcrateeq']['dndt'][i, j]])[0]
                dvdt = np.array(file[set['jpcrateeq']['dvdt'][i, j]])[0]
                dpdt = np.array(file[set['jpcrateeq']['dpdt'][i, j]])[0]
                resultPy = jpcrateeq(t, theta)
                np.testing.assert_allclose(np.reshape(resultPy[0], 1), dndt)
                np.testing.assert_allclose(np.reshape(resultPy[1], 1), dvdt)
                np.testing.assert_allclose(np.reshape(resultPy[2], 1), dpdt)
