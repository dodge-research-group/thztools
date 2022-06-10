import numpy as np
from thztools.thztoolsPY.noisevar import noisevar
from thztools.thztoolsPY.tdtf import tdtf

def costfun(fun, mu ,theta, xx, yy, sigma_alpha, sigma_beta, sigma_tau, ts):
    """
    costfun computes the MLE cost function

    Parameters
    ----------
        fun :

        mu :

        theta :

        xx :

        yy :

        sigma_alpha :

        sigma_beta :

        sigma_tau :

        ts :

    Returns
    -------
        K :

    """

    sigma = [sigma_alpha, sigma_beta, sigma_tau]

    n = len(xx)
    H = tdtf(fun, theta, n, ts)

    psi = H*mu

    Vmu = np.diag(noisevar(sigma, mu, ts))
    Vpsi = np.diag(noisevar(sigma, psi, ts))

    iVmu = np.diag(1/noisevar(sigma, mu, ts))
    iVpsi = np.diag(1/noisevar(sigma, psi, ts))

    # compute the inverse covariance matrices for xx and yy
    iVx = np.diag(1/noisevar(sigma, xx, ts))
    iVy = np.diag(1/noisevar(sigma, xx, ts))

    # compute the cost function
    # Note: sigmamu and sigmapsi both have determinatns below the numerical
    # precision, so we multiply them by the constant matrices isigmaxx and
    # isigmayy to improve numerical stability
    K = np.log(np.linalg.det(iVx*Vmu)) + np.log(np.linalge.det(iVy*Vpsi)) + \
        (xx - mu).T*iVmu*(xx - mu) + (yy - psi).T*iVpsi*(yy - psi)

    return K
