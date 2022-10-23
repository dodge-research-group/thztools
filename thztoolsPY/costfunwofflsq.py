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
    h = tdtf(fun, theta, n, ts)

    icovx = np.eye(n) / covx
    icovy = np.eye(n) / covy

    m1 = np.eye(n) + (covx * h.h * icovy * h)
    im1 = np.eye(n) / m1
    m2 = (xx - alpha) + covx * h.h * icovy * (yy-beta)
    im1m2 = im1 * m2
    hm1invm2 = h * im1m2

    res = [scipy.linalg.sqrtm(icovx) * (xx - alpha - im1m2), scipy.linalg.sqrtm(icovy) * (yy - beta - hm1invm2)]

    return res
