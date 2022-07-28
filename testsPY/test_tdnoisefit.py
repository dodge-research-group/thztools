import numpy as np
import h5py

from thztoolsPY.tdnoisefit import tdnoisefit


def test():
    with h5py.File('tdnoisefit_test_data.mat', 'r') as file:
        Set = file['tdnoisefit_data']
        x = file['tdnoisefit_data']['x']
        options = file['tdnoisefit_data']['Options']
        output = file['tdnoisefit_data']['Output']

        xn = np.transpose(np.array(x))
