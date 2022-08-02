import numpy as np
import scipy

from thztoolsPY.tdtf import tdtf


def costfunwofflsq(fun, theta, xx, yy, alpha, beta, covx, covy, ts):
    """

    Parameters
    ----------
    fun
    theta
    xx
    yy
    alpha
    beta
    covx
    covy
    ts

    Returns
    -------

    """
    n = len(xx)
    H = tdtf(fun, theta, n, ts)

    icovx = np.eye(n) / covx
    icovy = np.eye(n) / covy

    M1 = np.eye(n) + (covx * H.H * icovy * H)
    iM1 = np.eye(n) / M1
    M2 = (xx - alpha) + covx * H.H * icovy * (yy-beta)
    iM1M2 = iM1 * M2
    HM1invM2 = H * iM1M2

    res = [scipy.linalg.sqrtm(icovx) * (xx - alpha - iM1M2), scipy.linalg.sqrtm(icovy) * (yy - beta - HM1invM2)]

    return res
