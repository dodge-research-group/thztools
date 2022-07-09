import numpy as np
import h5py

from thztools.thztoolsPY.airscancorrect import airscancorrect

def test_airscancorrect():
    with h5py.File('airscancorrect_test_data.mat', 'r') as file:
        Set = file['Set']
        xdata = Set['airscancorrect']['x'][0]
        paramdata = Set['airscancorrect']['param'][0]
        dfx = np.array(xdata)

        for i in range(0,dfx.shape[0]):
            x = np.array(file[Set['airscancorrect']['x'][0, i]]).T
            param = {'A': np.array(file[paramdata[i]]['A'])[0],
                     'eta': np.array(file[paramdata[i]]['eta'])[0],
                     'ts': np.array(file[paramdata[i]]['ts'])[0]}
            Xadj = np.array(file[Set['airscancorrect']['Xadj'][0, i]]).T
            XadjPY = airscancorrect(x, param)
            np.testing.assert_allclose(XadjPY, Xadj)
