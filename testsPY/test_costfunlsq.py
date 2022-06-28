import numpy as np
import h5py

from thztoolsPY.costfunlsq import costfunlsq

with h5py.File('costfunlsq_test_data.mat', 'r') as file:
    Set = file['Set']
    xF = file['Set']['costfunlsq']['F'][0]
    xT = file['Set']['epswater']['T'][0]
    dfF = np.array(xF)
    dfT = np.array(xT)