import numpy as np
from scipy.optimize import minimize
from thztoolsPY.tdtf import tdtf
from thztoolsPY.tdnll import tdnll
import scipy


def tdnoisefit(x, param, fix={'logv': False, 'mu': False, 'a': True, 'eta': True}, ignore={'a': True, 'eta': True}):
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
           .exitflag    Exit flag from FMINUNC
           .output     Output from FMINUNC
           .grad        NLL cost function gradient from FMINUNC
           .hessian     NLL cost function hessian from FMINUNC
     """
    n, m = x.shape

    # Parse Inputs
    if 'v0' in param:
        v0 = param['v0']
    else:
        v0 = np.mean(np.var(x, 1)) * np.array([1, 1, 1])
        param['v0'] = v0

    if 'mu0' in param:
        mu0 = param['mu0']
    else:
        mu0 = np.mean(x, 1)
        param['mu0'] = mu0

    if 'a0' in param:
        a0 = param['a0']
    else:
        a0 = np.ones(m)
        param['a0'] = a0

    if 'eta0' in param:
        eta0 = param['eta0']

    else:
        eta0 = np.zeros(m)
        param['eta0'] = eta0

    if 'ts' in param:
        ts = param['ts']
    else:
        ts = 1
        param['ts'] = 1

    mle = {'x0': np.array([])}
    idxstart = 0
    idxrange = dict()

    if fix['logv']:
        def setplogv(p):
            return np.log(param['v0'])
    else:
        mle['x0'] = np.concatenate((mle['x0'], np.log(param['v0'])))
        idxend = idxstart + 3
        idxrange['logv'] = np.arange(idxstart, idxend)

        def setplogv(p):
            return p[idxrange['logv']]

        idxstart = idxend
    pass

    if fix['mu']:
        def setpmu(p):
            return param['mu0']
    else:
        mle['x0'] = np.concatenate((mle['x0'], param['mu0']))
        idxend = idxstart + n
        idxrange['mu'] = np.arange(idxstart, idxend)

        def setpmu(p):
            return p[idxrange['mu']]

        idxstart = idxend
    pass

    if ignore['a']:
        def setpa(p):
            return []

    elif fix['a']:
        def setpa(p):
            return param['a0']

    elif fix['mu']:
        mle['x0'] = np.concatenate((mle['x0'], param['a0']))
        idxend = idxstart + m
        idxrange['a'] = np.arange(idxstart, idxend)

        def setpa(p):
            return p[idxrange['a']]

        idxstart = idxend
    else:
        mle['x0'] = np.concatenate((mle['x0'], param['a0'][1:] / param['a0'][0]))
        idxend = idxstart + m - 1
        idxrange['a'] = np.arange(idxstart, idxend)

        def setpa(p):
            return np.concatenate(([1], p[idxrange['a']]), axis=0)

        idxstart = idxend
    pass

    if ignore['eta']:
        def setpeta(p):
            return []

    elif fix['eta']:
        def setpeta(p):
            return param['eta0']

    elif fix['mu']:
        mle['x0'] = np.concatenate((mle['x0'], param['eta0']))
        idxend = idxstart + m
        idxrange['eta'] = np.arange(idxstart, idxend)

        def setpeta(p):
            return p[idxrange['eta']]

    else:
        mle['x0'] = np.concatenate((mle['x0'], param['eta0'][1:] - param['eta0'][0]))
        idxend = idxstart + m - 1
        idxrange['eta'] = np.arange(idxstart, idxend)

        def setpeta(p):
            return np.concatenate(([0], p[idxrange['eta']]), axis=0)
    pass

    def fun(theta, w):
        return -1j * w

    d = tdtf(fun, 0, n, param['ts'])

    def parsein(p):
        return {'logv': setplogv(p), 'mu': setpmu(p), 'a': setpa(p), 'eta': setpeta(p), 'ts': param['ts'], 'd': d}

    def objective(p):
        return tdnll(x, parsein(p), fix)[0]

    def jacobian(p):
        return tdnll(x, parsein(p), fix)[1]

    mle['objective'] = objective
    out = minimize(mle['objective'], mle['x0'], method='BFGS', jac=jacobian)

    # The trust-region algorithm returns the Hessian for the next-to-last
    # iterate, which may not be near the final point. To check, test for
    # positive definiteness by attempting to Cholesky factorize it. If it
    # returns an error, rerun the optimization with the quasi-Newton algorithm
    # from the current optimal point.

    try:
        np.linalg.cholesky(np.linalg.inv(out.hess_inv))
        hess = np.linalg.inv(out.hess_inv)
    except:
        print('Hessian returned by FMINUNC is not positive definite;\n'
              'recalculating with quasi-Newton algorithm')

        mle['x0'] = out.x
        out2 = minimize(mle['objective'], mle['x0'], method='BFGS', jac=jacobian)
        hess = np.linalg.inv(out2.hess_inv)

    # Parse output
    p = {}
    idxrange = dict()
    idxstart = 0

    if fix['logv']:
        p['var'] = param['v0']
    else:
        idxend = idxstart + 3
        idxrange['logv'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['var'] = np.exp(out.x[idxrange['logv']])
    pass

    if fix['mu']:
        p['mu'] = param['mu0']
    else:
        idxend = idxstart + n
        idxrange['mu'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['mu'] = out.x[idxrange['mu']]
    pass

    if ignore['a'] or fix['a']:
        p['a'] = param['a0']
    elif fix['mu']:
        idxend = idxstart + m
        idxrange['a'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['a'] = out.x[idxrange['a']]
    else:
        idxend = idxstart + m - 1
        idxrange['a'] = np.arange(idxstart, idxend)
        idxstart = idxend
        p['a'] = np.concatenate(([1], out.x[idxrange['a']]), axis=0)
    pass

    if ignore['eta'] or fix['eta']:
        p['eta'] = param['eta0']
    elif fix['mu']:
        idxend = idxstart + m
        idxrange['eta'] = np.arange(idxstart, idxend)
        p['eta'] = out.x[idxrange['eta']]
    else:
        idxend = idxstart + m - 1
        idxrange['eta'] = np.arange(idxstart, idxend)
        p['eta'] = np.concatenate(([0], out.x[idxrange['eta']]), axis=0)
    pass

    p['ts'] = param['ts']

    varyParam = np.logical_not([fix['logv'], fix['mu'], fix['a'] or ignore['a'], fix['eta'] or ignore['eta']])
    diagnostic = {'grad': out.jac, 'hessian': hess,
                  'err': {'var': [], 'mu': [], 'a': [], 'eta': []}}
    v = np.dot(np.eye(hess.shape[0]), scipy.linalg.inv(hess))
    err = np.sqrt(np.diag(v))
    idxstart = 0
    if varyParam[0]:
        diagnostic['err']['var'] = np.sqrt(np.diag(np.diag(p['var']) * v[0:3, 0:3]) * np.diag(p['var']))
        idxstart = idxstart + 3
    pass

    if varyParam[1]:
        diagnostic['err']['mu'] = err[idxstart:idxstart + n]
        idxstart = idxstart + n
    pass

    if varyParam[2]:
        if varyParam[1]:
            diagnostic['err']['a'] = err[idxstart:idxstart + m - 1]
            idxstart = idxstart + m - 1
        else:
            diagnostic['err']['a'] = err[idxstart:idxstart + m]
            idxstart = idxstart + m
    pass

    if varyParam[3]:
        if varyParam[1]:
            diagnostic['err']['eta'] = err[idxstart:idxstart + m - 1]
        else:
            diagnostic['err']['eta'] = err[idxstart:idxstart + m]
    pass

    return [p, out.fun, diagnostic]
