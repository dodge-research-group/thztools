import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class DataPulse:

    def __init__(self, filename=''):
        self.AcquisitionTime = None
        self.Description = None
        self.TimeConstant = None
        self.WaitTime = None
        self.setPoint = None
        self.scanOffset = None
        self.temperature = None
        self.time = None
        self.amplitude = None
        self.ChannelAMaximum = None
        self.ChannelAMinimum = None
        self.ChannelAVariance = None
        self.ChannelASlope= None
        self.ChannelAOffset = None
        self.ChannelBMaximum = None
        self.ChannelBMinimum = None
        self.ChannelBVariance = None
        self.ChannelBSlope = None
        self.ChannelBOffset = None
        self.ChannelCMaximum = None
        self.ChannelCMinimum = None
        self.ChannelCVariance = None
        self.ChannelCSlope = None
        self.ChannelCOffset = None
        self.ChannelDMaximum = None
        self.ChannelDMinimum = None
        self.ChannelDVariance = None
        self.ChannelDSlope = None
        self.ChannelDOffset = None
        self.dirname = None
        self.file = None
        self.filename = filename
        self.frequency = None

        if filename is not None:

            data = pd.read_csv(filename, header=None, delimiter='\t')


            for i in range(data.shape[0]):
                try:
                    float(data[0][i])
                except:
                    ind = i + 1
                pass

            keys = data[0][0:ind].to_list()
            vals = data[1][0:ind].to_list()

            for k in range(len(keys)):
                if keys[k] in list(self.__dict__.keys()):

                    try:
                        float(vals[k])
                        setattr(self, keys[k], float(vals[k]))
                    except ValueError:
                        setattr(self, keys[k], vals[k])


                else:
                    raise Warning('The data key is not defied')


            self.time = data[0][ind:].to_numpy(dtype=float)
            self.amplitude = data[1][ind:].to_numpy(dtype=float)


            # Calculate frequency range
            self.frequency = (np.arange(np.floor(len(self.time))) / 2 - 1).T / (self.time[-1] - self.time[0])

            # Calculate fft
            famp = np.fft.fft(self.amplitude)
            self.famp = famp[0:int(np.floor(len(famp)/2))]