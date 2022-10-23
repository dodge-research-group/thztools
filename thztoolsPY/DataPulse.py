import os
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime


class DataPulse:
    """
    DataPulse is a class which can import scan data from an appropriately formatted file and puts both the numerical and
    any supporting data into class properties-


    """

    def __init__(self, acquisitionTime=datetime.fromtimestamp(0), timeConstant=math.nan, waitTime=math.nan, description='',
                 setPoint=math.nan, scanOffset=math.nan, temperature=math.nan, time=math.nan, amplitude=math.nan,
                 channelAMaximum=math.nan, channelAMinimum=math.nan, channelAVariance=math.nan, channelASlope=math.nan,
                 channelAOffset=math.nan, channelBMaximum=math.nan, channelBMinimum=math.nan, channelBVariance=math.nan,
                 channelBSlope=math.nan, channelBOffset=math.nan, channelCMaximum=math.nan, channelCMinimum=math.nan,
                 channelCVariance=math.nan, channelCSlope=math.nan, channelCOffset=math.nan, channelDMaximum=math.nan,
                 channelDMinimum=math.nan, channelDVariance=math.nan, channelDSlope=math.nan, channelDOffset=math.nan,
                 dirname='', file={'name': '', 'data': '', 'bytes': 0, 'isdir': False, 'datenum': 0}):
        self.acquisitionTime = acquisitionTime
        self.timeConstant = timeConstant
        self.waitTime = waitTime
        self.description = description
        self.setPoint = setPoint
        self.scanOffset = scanOffset
        self.temperature = temperature
        self.time = time
        self.amplitude = amplitude
        self.channelAMaximum = channelAMaximum
        self.channelAMinimum = channelAMinimum
        self.channelAVariance = channelAVariance
        self.channelASlope= channelASlope
        self.channelAOffset = channelAOffset
        self.channelBMaximum = channelBMaximum
        self.channelBMinimum = channelBMinimum
        self.channelBVariance = channelBVariance
        self.channelBSlope = channelBSlope
        self.channelBOffset = channelBOffset
        self.channelCMaximum = channelCMaximum
        self.channelCMinimum = channelCMinimum
        self.channelCVariance = channelCVariance
        self.channelCSlope = channelCSlope
        self.channelCOffset = channelCOffset
        self.channelDMaximum = channelDMaximum
        self.channelDMinimum = channelDMinimum
        self.channelDVariance = channelDVariance
        self.channelDSlope = channelDSlope
        self.channelDOffset = channelDOffset
        self.dirname = dirname
        self.file = file

    @property
    def frequency(self):
        # Calculate frequency range
        frequency = (np.arange(np.floor(len(self.time))) / 2 - 1).T / (self.time[-1] - self.time[0])
        return frequency

    @property
    def fourierAmplitude(self):
        # Calculate fft
        famp = np.fft.fft(self.amplitude)
        famp = famp[0:np.floor(len(famp) / 2)]
        return famp

    def DataPulse(self, filename):
        obj = dict()
        [obj['dirname'], obj['File']] = os.path.split(filename) # initialize from a datafile or as an empty pulse
        fid = open(filename)
        # first get the time the scan was taken, which appears on
        # the first line of the file
        lines = fid.readlines()
        date_time_str = lines[0].split()[1]
        obj['AcquisitionTime'] = datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%S')
        # now keep grabbing each line of the file and matching it to
        # the property of the same name
        #print('lines', lines)
        d1 = np.array(lines)
        for idx, strr in enumerate(lines[1:]):
            if strr == '\n':
                break
            else:
                var_name = strr.split()[0]
                var_val = strr.split()[1]

                if len(var_name)==2:
                    obj[var_name[0]] = float(var_val[1])


        # store time and amplitude measurements

        obj['time'] = []
        obj['amp'] = []

        for i in lines[(idx+3):]:

            # get data in each line of the file data. It generates a class list
            data_list = i.split()

            # convert each element on the list into a string
            data_time_str = ''.join(data_list[0:1])
            data_amp_str = ''.join(data_list[1:2])

            # convert string into float and save in a usual list
            data_time = (np.float(data_time_str))
            data_amp = np.array(np.float(data_amp_str))
            obj['time'].append(data_time)
            obj['amp'].append(data_amp)

        fid.close()
        obj['time'] = np.array(obj['time'])
        obj['amp'] = np.array(obj['amp'])

        return obj

    def plot(self, obj, varargin):
        # Plots the time trace and power spectrum
        plt.figure()
        for o in obj:
            plt.plot(o.time, o.amplitude)
        plt.xlabel('Time (ps)')
        plt.ylabel('Voltage (V)')
        plt.figure()
        for o in obj:
            plt.plot(o.frequency, 10 * np.log10(np.abs(o.fourierAmplitude)**2), varargin)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Power (dB)')
