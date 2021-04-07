import numpy as np
import datetime


class Thzwaveform:
    def __init__(self, filename):
        self.filename = filename
        self.local_attributes = {}
        thz_file = open(self.filename, 'r')
        for line in thz_file:
            attributes = line.split("\t")
            if attributes[0].isalpha():
                self.local_attributes[attributes[0]] = \
                    attributes[1].rstrip("\n")
                if str(attributes[0]) == 'AcquisitionTime':
                    self.Acquisition_time = date_time_obj = \
                        datetime.datetime.strptime(attributes[1].rstrip("\n"),
                                                   '%Y-%m-%dT%H:%M:%S')
                if str(attributes[0]) == 'WaitTime':
                    self.Wait_time = float(attributes[1].rstrip("\n"))
                if str(attributes[0]) == 'TimeConstant':
                    self.Time_constant = float(attributes[1].rstrip("\n"))
            else:
                self.time, self.amplitude = np.genfromtxt('Scan.thz', 
                                                          comments='#',
                                                          unpack=True,
                                                          dtype=None,
                                                          skip_header=3,
                                                          encoding=None)

                                        
class Thzwaveform2:
    def _init__(self, filename, local_attributes, Acquisition_time, Wait_time,
                Time_constant, time, amplitude):
        self.filename = filename
        self.local_attributes = local_attributes
        self.Acquisition_time = Acquisition_time
        self.Wait_time = Wait_time
        self.Time_constant = Time_constant
        self.time = time
        self.amplitude = amplitude

    def Thzread(self, filename):
        self.filename = filename
        local_attributes = {}
        thz_file = open(filename, 'r')
        for line in thz_file:
            attributes = line.split("\t")
            if attributes[0].isalpha():
                local_attributes[attributes[0]] = attributes[1].rstrip("\n")
                if str(attributes[0]) == 'AcquisitionTime':
                    Acquisition_time = date_time_obj = \
                        datetime.datetime.strptime(attributes[1].rstrip("\n"),
                                                   '%Y-%m-%dT%H:%M:%S')
                if str(attributes[0]) == 'WaitTime':
                    Wait_time = float(attributes[1].rstrip("\n"))
                if str(attributes[0]) == 'TimeConstant':
                    Time_constant = float(attributes[1].rstrip("\n"))
            else:
                time, amplitude = np.genfromtxt('Scan.thz', comments='#',
                                                unpack=True, dtype=None,
                                                skip_header=3, encoding=None)
