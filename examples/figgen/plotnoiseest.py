import os
import numpy as np
from matplotlib import pyplot as plt
from thztools.thztools import DataPulse
from calcnoise import calcnoise

# get the data from path
path = './../data/2017-03-20/Air Scans/Normal'

# using DataPulse class get the amplitude
air_scans = []
for file in os.listdir(path):
    obj = DataPulse(path + '/' + file).amplitude
    air_scans.append(obj)

# get the time array
t = DataPulse('./../data/2017-03-20/Air Scans/Normal/Scan003.thz').time

# matrix with amplitude vectors
x = np.vstack(air_scans).T


gain = 5e7
x = x * 1e12 / gain

# noise model
data = calcnoise(t, x)

# get the data to plot
t = data['t'][0:256]
vtot = data['vtot'][0:256]
xadjusted = data['xadjusted'][:256, :]
n, m = xadjusted.shape


# plotnoisedata
plt.figure(figsize=(14, 8))
plt.plot(t, xadjusted[:, 1:-1])
plt.plot(t, xadjusted[:, -1], color='red', linewidth=1)
plt.xlabel('t (ps)', fontsize=16)
plt.ylabel(r'$\hat{\sigma}_{\mu}^{*}$ (pA)', fontsize=16)
plt.show()


# plotnoiseseest
plt.figure(figsize=(14, 8))
plt.scatter(t, np.std(xadjusted, 1, ddof=1), color='black')
plt.plot(t, np.sqrt(vtot) * m / (m - 1), color='red')
plt.xlabel('t (ps)', fontsize=16)
plt.ylabel(r'$\hat{\sigma}_{\mu}^{*}$ (pA)', fontsize=16)
plt.show()


delta = data['delta'][0:256]

# plotnoiseres
plt.figure(figsize=(14, 8))
plt.stem(t, delta[:, 5] / np.sqrt(m / (m - 1)))
plt.xlabel('t (ps)', fontsize=16)
plt.ylabel(r'$\hat{\sigma}_{\mu}^{*}$ (pA)', fontsize=16)
plt.show()
