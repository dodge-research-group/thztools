import numpy as np

def tdnoisefit(x, param, fix = {'logv': False, 'mu': False, 'a': True, 'eta': True}, ignore={'a': True, 'eta': True}, *args):

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


    mle = {}
    idxstart = 1

    if fix['logv']:

        def setplogv(p):
            return np.log(param['v0'])
    else:
        mle['x0'] = [np.log(param['v0'])]
        idxend = idxstart + 2
        idxrange = np.arange(idxstart, idxend + 1)

        def setplogv(p):
            return

        idzstart = idxend + 1
    pass

    ### help here ###################3
    if fix['mu']:
        def setpmu(p):
            return mu0
    else:
        mle['x0'] = [param['mu0']]
        idxend = idxstart + n - 1
        idxrange = np.arange(idxstart, idxend + 1)

        def setpmu(p):
            return p(idxrange)

        idxstart = idxend + 1
    pass
    ###########################

    if ignore['a']:
        def setpa(p):
            return = []

    elif:
        def setpa(p):
            return param['a0']

    else:
        mle['x0'] = param['a0']
        idxend = idxstart + m - 1