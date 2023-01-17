import numpy as np

from thztoolsPY.noisevar import noisevar
from thztoolsPY.tdtf import tdtf


def costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, sigma_tau, ts):
    """
    Computes the maximum likelihood cost function.

    Parameters
    ----------
        fun : callable
            Transfer function, in the form fun(theta,w), -iwt convention.

        mu : ndarray
            Signal vector of size (n,).

        theta : ndarray
            Input parameters for the function.

        xx : ndarray
            Measured input signal.

        yy : ndarray
            Measured output signal.

        sigma_alpha : float
            Noise produced by the detection electronics. ($\\ \sigma _{\\alpha} = 10^{-4} $)

        sigma_beta : float
            Multiplicative noise term produced by laser power fluctuations. ($\\ \sigma _{\\beta} = 10^{-4} $)

        sigma_tau : float
            Noise produced by fluctuations in the optical path lengths.

        ts : float
            Sampling time.

    Returns
    -------
        k : callable
            Negative loglikelihood function.

    """

    sigma = [sigma_alpha, sigma_beta, sigma_tau]

    n = len(xx)
    h = tdtf(fun, theta, n, ts)

    psi = h*mu

    vmu = np.diag(noisevar(sigma, mu, ts))
    vpsi = np.diag(noisevar(sigma, psi, ts))

    ivmu = np.diag(1/noisevar(sigma, mu, ts))
    ivpsi = np.diag(1/noisevar(sigma, psi, ts))

    # compute the inverse covariance matrices for xx and yy
    ivx = np.diag(1/noisevar(sigma, xx, ts))
    ivy = np.diag(1/noisevar(sigma, xx, ts))

    # compute the cost function
    # Note: sigma-mu and sigma-psi both have determinants below the numerical
    # precision, so we multiply them by the constant matrices isigmaxx and
    # isigmayy to improve numerical stability
    k = np.log(np.linalg.det(ivx*vmu)) + np.log(np.linalge.det(ivy*vpsi)) + \
        (xx - mu).T*ivmu*(xx - mu) + (yy - psi).T*ivpsi*(yy - psi)

    return k

