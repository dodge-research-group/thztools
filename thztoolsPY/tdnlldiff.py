import numpy as np

from thztools.thztoolsPY.tdtf import tdtf

def tdnlldiff(x, param, *args):

    # Parse function inputs
    [N, M] = x.shape
    if len(args) > 2:
        Fix = args[0]
    else:
        Fix = {'logv':False, 'mu':False, 'A':False, 'eta':False}

    # Parse parameter dictionary
    Pfields = param.keys()
    Ignore = {}
    if 'logv' in Pfields:
        v = np.exp(param.get('logv'))
    else:
        # error('TDNLL requires Param structure with logv field')
        pass
    if 'mu' in Pfields:
        mu = (param.get('mu'))
    else:
        # error('TDNLL requires Param structure with mu field')
        pass
    if ('A' in Pfields) and (param.get('A') != None):
        A = param.get('A')
        Ignore['A'] = False
    else:
        A = np.ones((M,1))
        Ignore['A'] = True
    if ('eta' in Pfields) and (param.get('eta') != None):
        eta = param.get('eta')
        Ignore['eta'] = False
    else:
        eta = np.zeros((M,1))
        Ignore['eta'] = True
    if 'ts' in Pfields:
        ts = param.get('ts')
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')
    if 'D' in Pfields:
        D = param.get('D')
    else:
        pass # work on fun and including tdtf

    # Compute frequency vector and Fourier coefficients of mu

    pass
