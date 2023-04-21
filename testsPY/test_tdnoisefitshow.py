import h5py
import numpy as np
from thztoolsPY.noisefitshow import noisefitshow
import matplotlib.pyplot as plt


def test():

    file = h5py.File('dataToptica.mat', 'r')

    # time
    t = np.squeeze(np.array(file['dataToptica']['t']))

    # Data for 50 ps, 100 avg

    x1 = np.transpose(np.array(file['dataToptica']['X1']))
    a1 = np.array(file['dataToptica']['A1'])
    mu1 = np.array(file['dataToptica']['mu1'])
    v1 = np.array(file['dataToptica']['v1'])
    eta1 = np.array(file['dataToptica']['eta1'])
    xadjusted_matlab1 = np.array(file['dataToptica']['Xadjusted1'])
    sigmatotstar_matlab1 = np.array(file['dataToptica']['sigmaTotstar1'])

    # data from python
    out1 = noisefitshow(t, x1)

    # data from Matlab
    sigmatotstar_matlab1 = np.array(file['dataToptica']['sigmaTotstar1'])

    np.testing.assert_allclose(out1['sigmatotstar'], np.squeeze(sigmatotstar_matlab1.T))

    # plot python and matlab fit
    #plt.figure(figsize=(10, 8))
    #plt.title('Noise Model Results, 50 ps 2 avgs', fontsize=15)
    # python
    #plt.plot(t, out1['sigmatotstar'], color='red', label='Python', linewidth=2)
    #plt.plot(t, np.squeeze(sigmatotstar_matlab1.T), color='blue', label='Matlab', linewidth=2)
    #plt.xlabel('Time (ps)', fontsize=15)
    #plt.ylabel('$\sigma \hat{} ^{*}$ (nA)', fontsize=15)
    #plt.legend(fontsize=15)
    #plt.show()
