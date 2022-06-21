import numpy as np

from thztools.thztoolsPY.tdnll import tdnll
from thztools.thztoolsPY.tdtf import tdtf


def tdnoisefit(x, param, fix = {'logv': False, 'mu': False, 'a': True, 'eta': True}, ignore={'a': True, 'eta': True}):

    """ TDNOISEFIT computes MLE parameters for the time-domain noise model

     Syntax:   P = tdnoisefit(x,Oxptions)

     Description:

     TDNOISEFIT computes the noise parameters sigma and the underlying signal
     vector mu for the data matrix x, where the columns of x are each noisy
     measurements of mu.

     Inputs:
       x               Data matrix

     Optional inputs:
       Options         Fit options

     Option fields:
       v0              Initial guess, noise model parameters [3x1 double]
       mu0             Initial guess, signal vector [Nx1 double]
       A0              Initial guess, amplitude vector [Mx1 double]
       eta0            Initial guess, delay vector [Mx1 double]
       ts              Sampling time [scalar double]
       Fix             Fixed variables [struct]
       Ignore          Ignore variables [struct]

     Outputs:
       P               Output parameter structure
           .logv       Log of noise parameters
           .mu         Signal vector
           .A          Amplitude vector
           .eta        Delay vector
           .ts         Samling time
       fval            Value of NLL cost function from FMINUNC
       Diagnostic      Structure of diagnostic information
           .exitflag	Exit flag from FMINUNC
           .output     Output from FMINUNC
           .grad     	NLL cost function gradient from FMINUNC
           .hessian   	NLL cost function hessian from FMINUNC
     """


    n, m = x.shape
    mle = {}
    idxstart = 0

    if fix['logv']:

        def setplogv(p):
            return np.log(param['v0'])
    else:
        mle['x0'] = [np.log(param['v0'])]
        idxend = idxstart + 2
        idxrange = np.arange(idxstart, idxend + 1)

        def setplogv(p):
            return p[idxrange]

        idxstart = idxend + 1
    pass


    if fix['mu']:
        def setpmu(p):
            return param['mu0']
    else:
        mle['x0'] = [param['mu0']]
        idxend = idxstart + n - 1
        idxrange = np.arange(idxstart, idxend + 1)

        def setpmu(p):
            return p[idxrange]

        idxstart = idxend + 1
    pass


    if ignore['a']:
        def setpa(p):
            return p[idxrange]

    elif fix['a']:
        def setpa(p):
            return param['a0']

    else:
        mle['x0'] = param['a0']
        idxend = idxstart + m - 1
        idxrange = np.arange(idxstart, idxend+1)
        def setpa(p):
            return p[idxrange]
        idxstart = idxend + 1
    pass

    if ignore['eta']:
        setpeta = param['eta']

    else:
        mle['x0'] = [param['eta0']]
        idxend = idxstart + m - 1
        idxrange = np.arange(idxstart, idxend + 1)
        def setpeta(p):
            return p(idxrange)
    pass

    def fun(theta, w):
        return -1j*w

    d = tdtf(fun, 0, n, param['ts'])

    def parsein(p):
        return {'logv': setplogv(p), 'mu': setpmu(p), 'a': setpa(p), 'eta': setpeta(p), 'ts': param['ts'], 'd': d}

    def objective(p):
        return tdnll(x, parsein(p), fix)

    mle['objective'] = objective

##############
    p = {}

    idxstart = 0
    if fix['logv']:
        p['var'] = param['v0']

    else:
        idxend = idxstart + 3
        idxrange = np.arange(idxstart, idxend + 1)
        p['var'] = np.exp
        ################
    pass

    if fix['mu']:
        p['mu'] = param['mu0']

    else:
        idxend = idxstart + n - 1
        idxrange = np.arange(idxstart, idxend + 1)
        idxstart = idxend + 1
        #######################
    pass

    if ignore['a'] or fix['a']:
        p['a'] = param['a0']

    else:
        idxend = idxstart + m - 1
        idxrange = np.arange(idxstart, idxend + 1)
        idxstart = idxend + 1
        ##################
    pass

    if ignore['eta'] or fix['eta']:
        p['eta'] = param['eta0']
        ######################################
    pass

    p['ts'] = param['ts']






