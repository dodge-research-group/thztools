import os

import numpy as np

from thztoolsPY.DataPulse import DataPulse


def thzload(varargin):
    dir_name = varargin['dirName']
    extension = varargin['extension']
    exclude = varargin['exclude']
    input_data = varargin['InputData']

    ListLoc = os.listdir(dir_name)
    nList = len(ListLoc)
    # DataLoc = DataPulse.empty()

    i_data_loc = 0
    for iFile in np.arange(nList):

    return data