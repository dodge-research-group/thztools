import os

import h5py
import numpy as np
from thztoolsPY.noisefitshow import noisefitshow
import matplotlib.pyplot as plt


def test():
    fname = os.path.join(os.path.dirname(__file__), 'dataToptica.mat')
    file = h5py.File(fname, 'r')

    # time
    t = np.squeeze(np.array(file['dataToptica']['t']))

    # Data for 50 ps, 2 avg

    x1 = np.transpose(np.array(file['dataToptica']['X1']))
    out1 = noisefitshow(t, x1)

    # data from Matlab
    sigmatotstar_matlab1 = np.array(file['dataToptica']['sigmaTotstar1'])

    # test
    np.testing.assert_allclose(out1['sigmatotstar'], np.squeeze(sigmatotstar_matlab1.T), atol= 1e-3, rtol = 1e-3)


# plot python and matlab fit
#    plt.figure(figsize=(10, 8))
#       plt.title('Noise Model Results, 50 ps 2 avgs', fontsize=15)
# python
#     plt.plot(t, out2['sigmatotstar'], color='red', label='Python', linewidth=2)
#    plt.plot(t, np.squeeze(sigmatotstar_matlab2.T), color='blue', label='Matlab', linewidth=2)
#   plt.xlabel('Time (ps)', fontsize=15)
#   plt.ylabel('$\sigma \hat{} ^{*}$ (nA)', fontsize=15)
#   plt.legend(fontsize=15)
#   plt.show()
