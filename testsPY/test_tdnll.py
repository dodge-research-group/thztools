import numpy as np
import h5py

from thztoolsPY.tdnll import tdnll

def test():
    with h5py.File('tdnll_test_data.mat', 'r') as file: