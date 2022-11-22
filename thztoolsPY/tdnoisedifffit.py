import numpy as np
from scipy.optimize import minimize

from thztoolsPY.tdnll import tdnll
from thztoolsPY.tdtf import tdtf


def tdnoisefffit(x, param, fix= {'logv': False, 'mu': False, 'a': True, 'eta': True}, ignore={'a': True, 'eta': True}):
    """ Computes maximum likelihood parameters for the time-domain noise model.

     Tdnosefffit computes the noise parameters sigma and the underlying signal
     vector mu for the data matrix x, where the columns of x are each noisy
     measurements of mu.

    Parameters
    ----------

    x : ndarray or matrix
        Data matrix.

    param : dict
        A dictionary containing initial guess for the parameters:

        v0 : ndarray
            Initial guess, noise model parameters. Array of real elements of size (3, )

        mu0 : ndarray
            Initial guess, signal vector. Array of real elements of size  (n, )

        a0 : ndarray
            Initial guess, amplitude vector. Array of real elements of size (m, )

        eta0 : ndarray
            Initial guess, delay vector. Array of real elements of size (m, )

        ts : int
            Sampling time

    fix : dict, optional
        Fixed variables. If not given, chosen to set free all the variables.

    ignore : dict, optional
        Ignore variables. If not given, chosen to ignore both amplitude and delay.

    Returns
    --------
    p : dict
        Output parameter dictionary containing:
            eta : ndarray
                Delay vector.

            a : ndarray
                Amplitude vector.

            mu : ndarray
                Signal vector.

            var : ndarray
                Log of noise parameters


        fval : float
           Value of NLL cost function from FMINUNC

        Diagnostic : dict
            Dictionary containing diagnostic information
                err : dic
                    Dictionary containing  error of the parameters.

                grad : ndarray
                      Negative loglikelihood cost function gradient from scipy.optimize.minimize BFGS method.

                hessian : ndarray
                    Negative loglikelihood cost function hessian from scipy.optimize.minimize BFGS method.

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
            return []

    elif fix['a']:
        def setpa(p):
            return param['a0']

    else:
        mle['x0'] = param['a0'][1:]
        idxend = idxstart + m - 2
        idxrange = np.arange(idxstart, idxend + 1)

        def setpa(p):
            return p[idxrange]

        idxstart = idxend + 1
    pass

    if ignore['eta']:
        def setpeta(p):
            return []

    elif fix['eta']:
        def setpeta(p):
            return param['eta']

    else:
        mle['x0'] = [param['eta0']][1:]
        idxend = idxstart + m - 1
        idxrange = np.arange(idxstart, idxend + 1)

        def setpeta(p):
            return np.concatenate((np.array([param['eta']][0]), p(idxrange)))
    pass

    def fun(theta, w):
        return -1j * w

    d = tdtf(fun, 0, n, param['ts'])

    def parsein(p):
        return {'logv': setplogv(p), 'mu': setpmu(p), 'a': setpa(p), 'eta': setpeta(p), 'ts': param['ts'], 'd': d}

    def tdnllfixedeffects(x, param, fix):
        """ Strip first element of gradient vector wrt A and eta """

        nll, gradnll = tdnll(x, param, fix)
        idxstart = 0
        if np.logical_not(fix['logv']):
            idxstart = idxstart + 3
        pass

        if np.logical_not(fix['mu']):
            idxstart = idxstart + n
        pass

        if np.logical_not(fix['a']):
            gradnll[idxstart] = []
            idxstart = idxstart + m - 1
        pass

        if np.logical_not(fix['eta']):
            gradnll[idxstart] = []
        pass

        return nll, gradnll

    def objective(p):
        return tdnllfixedeffects(x, parsein(p), fix)

    mle['objective'] = objective

    out = minimize(mle['objective'], mle['x0'], method='BFGS')

    # Parse output
    p = {}

    idxstart = 0
    if fix['logv']:
        p['var'] = param['v0']

    else:
        idxend = idxstart + 3
        idxrange = np.arange(idxstart, idxend + 1)
        p['var'] = np.exp(out.x[idxrange])
    pass

    if fix['mu']:
        p['mu'] = param['mu0']

    else:
        idxend = idxstart + n - 1
        idxrange = np.arange(idxstart, idxend + 1)
        idxstart = idxend + 1
        p['mu'] = out.x[idxrange]
    pass

    if ignore['a'] or fix['a']:
        p['a'] = param['a0']

    else:
        idxend = idxstart + m - 2
        idxrange = np.arange(idxstart, idxend + 1)
        idxstart = idxend + 1
        p['a'] = np.concatenate((np.array([param['a0']][0]), out.x(idxrange)))
    pass

    if ignore['eta'] or fix['eta']:
        p['eta'] = param['eta0']

    else:
        idxend = idxstart + m - 2
        idxrange = np.array(idxstart, idxend + 1)
        p['eta'] = np.concatenate((np.array([param['eta']][0]), out.x(idxrange)))
    pass

    p['ts'] = param['ts']

    diagnostic = {'grad': out.jac, 'hessian': np.linalg.inv(out.hess_inv),
                  'err': {'var': [], 'mu': [], 'a': [], 'eta': []}}

    paramtest = np.logical_not([fix['logv'], fix['mu'], fix['a'] or ignore['a'], fix['eta'] or ignore['eta']])

    ngrad = np.sum(paramtest * [[3], [n], [m - 1], [m - 1]])
    v = np.identity(ngrad) / diagnostic['hessian']
    err = np.sqrt(np.diag(v))

    idxstart = 0
    if paramtest[0]:
        diagnostic['err']['var'] = np.sqrt(np.diag(np.diag(p['var']) * v[0:3, 0:3]) * np.diag(p['var']))
        idxstart = idxstart + 3
    pass

    if paramtest[1]:
        diagnostic['err']['mu'] = err[idxstart, (idxstart + n - 1)]
        idxstart = idxstart + n
    pass

    if paramtest[2]:
        diagnostic['err']['a'] = err[idxstart, (idxstart + m - 2)]
        idxstart = idxstart + m - 1
    pass

    if paramtest[3]:
        diagnostic['err']['eta'] = err(idxstart, (idxstart + m - 2))
    pass

    return p, out.fun, diagnostic
