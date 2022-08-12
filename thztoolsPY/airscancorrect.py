import numpy as np

from thztoolsPY.shiftmtx import shiftmtx


def airscancorrect(x, param):
    """
    AIRSCANCORRECT rescales and shifts data matrix

    Syntax:   Xadj = airscancorrect(X,Param)

    Description:

    AIRSCANCORRECT rescales and shifts each column of the data matrix X,
    assuming that each column is related to a common signal by an amplitude A
    and a delay eta.

    Parameters
    ----------
    x                   Data matrix
    param               Parameter dictionary, including:
                            A      Amplitude vector        [Mx1 double]
                            eta    Delay vector            [Mx1 double]
                            ts     Sampling time           [double]
    Returns
    -------
    Xadj                Adjusted data matrix
    """

    # Parse function inputs
    [N, M] = x.shape

    # Parse parameter structure
    Pfields = param.keys()
    if 'a' in Pfields and param.get('a') is not None:
        A = param.get('a').T
        # validateattributes(A, {'double'}, {'vector', 'numel', M})
        # Ignore.A = false
    else:
        A = np.ones((M, 1))
        # Ignore.A = true
    if 'eta' in Pfields and param.get('eta') is not None:
        eta = param.get('eta')
        # validateattributes(eta, {'double'}, {'vector', 'numel', M})
        # Ignore.eta = false
    else:
        eta = np.zeros((M, 1))
        # Ignore.eta = true
    if 'ts' in Pfields:
        ts = param['ts']
        # validateattributes(ts, {'double'}, {'scalar'})
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')

    Xadj = np.zeros((N, M))
    for m in np.arange(M):
        S = shiftmtx(-eta[m], N, ts)
        Xadj[:, m] = S @ (x[:, m] / A[m])

    return Xadj
