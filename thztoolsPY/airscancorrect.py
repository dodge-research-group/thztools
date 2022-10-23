import numpy as np
from thztoolsPY.shiftmtx import shiftmtx


def airscancorrect(x, param):
    """Airscancorrect rescales and shifts each column of the data matrix x,
    assuming that each column is related to a common signal by an amplitude A
    and a delay eta.

    Parameters
    ----------
    x : ndarray or matrix
        (n,m) data matrix


    param: dict
        Parameter dictionary including:
            A : ndarray
                (m, ) array containing amplitude vector
            eta : ndarray
                (m, ) array containing delay vector
            ts : float
                sampling time



    Returns
    -------
    xadj : ndarray or matrix
        Adjusted data matrix

    """

    # Parse function inputs
    [n, m] = x.shape

    # Parse parameter structure
    pfields = param.keys()
    if 'a' in pfields and param.get('a') is not None:
        a = param.get('a').T
    else:
        a = np.ones((m, 1))
        # Ignore.A = true
    if 'eta' in pfields and param.get('eta') is not None:
        eta = param.get('eta')
    else:
        eta = np.zeros((m, 1))
    if 'ts' in pfields:
        ts = param['ts']
    else:
        ts = 1

    xadj = np.zeros((n, m))
    for i in np.arange(m):
        s = shiftmtx(-eta[i], n, ts)
        xadj[:, i] = s @ (x[:, i] / a[i])

    return xadj
