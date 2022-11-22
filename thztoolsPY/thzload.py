import os
import numpy as np
from thztoolsPY.DataPulse import DataPulse


def thzload(varargin):
    """
    TH Recursively loads all terahertz files in a directory
        DATA = THZLOAD recursively loads all files with the .thz extension from
        the present directory and all of its subdirectories into DATA, an array
        of DataPulse objects

        DATA = THZLOAD(DIR) loads from the directory DIR

        DATA = THZLOAD(DIR, EXT) loads all files with the extension EXT

        DATA = THZLOAD(DIR, EXT, EXCLUDE) excludes directories that contain any
        strings in the array EXCLUDE (which may also be a character array or a
        cell array of character vectors)

        DATA = THZLOAD(DIR, EXT, EXCLUDE, DATAIN) appends data loaded from DIR
        to the input DataPulse array DATAIN

    Parameters
    ----------
    varargin

    Returns
    -------

    """
    dir_name = varargin['dirName']
    extension = varargin['extension']
    exclude = varargin['exclude']
    input_data = varargin['InputData']

    ListLoc = os.listdir(dir_name)
    nList = len(ListLoc)
    data_loc = []

    i_data_loc = 0
    for iFile in np.arange(nList):
        fname = ListLoc[iFile]
        if not os.path.isdir(fname) and fname.endswith(extension):
            data_loc[i_data_loc] = DataPulse(os.path.join(dir_name, fname))
        elif os.path.isdir(fname) and fname is not '.' and fname is not '..' and (exclude not in fname or not exclude):
            data_loc = thzload(os.path.join(dir_name, fname), extension, exclude, data_loc)

    data = [np.reshape(input_data, (len(input_data), 1)), np.reshape(data_loc, (len(data_loc), 1))]
    return data
