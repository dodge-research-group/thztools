import numpy as np

from thztoolsPY.fftfreq import fftfreq


def tdloglikelihood(param, x, ts):
    """
    Computes log-likelihood for the time-domain noise model.

    Tdloglikelihood computes the log-likelihood function for obtaining the
    data matrix x, given the parameter vector param.

    Parameters
    ----------
    x : ndarray or matrix
        Data matrix.

    param : ndarray
        A parameter vector including:

        logv : ndarray
            Log of noise parameters.

        mu : ndarray
            Signal vector of size (n,).

        loga : ndarray
            Log of amplitude vector of size (m,).

        eta : ndarray
            Delay vector of size (m,).

    ts : float
        Sampling time.

    Returns
    -------
    nll : callable
        Negative log-likelihood function

    gradnll : ndarray
        Gradient of the negative log-likelihood function
    """
    # Parse function inputs
    [n, m] = x.shape

    logv = param[:3]
    v = np.exp(logv)
    mu = param[3:(3 + n)]
    loga = param[(3 + n):(3 + n + m)]
    a = np.exp(loga)
    eta = param[(3 + n + m):(3 + n + 2 * m)]

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(n, ts)
    w = 2 * np.pi * f
    mu_f = np.fft.fft(mu)

    # Compute zeta
    exp_iweta = np.exp(1j * np.tile(w, m) * np.conj(np.tile(eta, n)).T)
    zeta_f = np.conj(np.tile(a, n)).T * np.conj(exp_iweta) * np.tile(mu_f, m)
    zeta = np.real(np.fft.ifft(zeta_f))

    # Compute log - likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    dzeta = np.real(np.fft.ifft(1j * np.tile(w, m) * zeta_f))

    valpha = v[0]
    vbeta = v[1] * zeta**2
    vtau = v[2] * dzeta ** 2
    vtot = valpha + vbeta + vtau

    resnormsq = ressq / vtot
    loglik = -m * n * np.log(2 * np.pi) / 2 - np.sum(np.log(vtot)) / 2 - np.sum(resnormsq) / 2

    # Compute gradient
    ngrad = 3 + n + 2 * m
    gradnll = np.zeros((ngrad, 1))

    reswt = res / vtot
    dvar = (vtot - ressq) / vtot**2

    # Gradient wrt logv
    gradnll[0] = -(1 / 2) * np.sum(dvar) * v[0]
    gradnll[1] = -(1 / 2) * np.sum(np.reshape(zeta, (len(zeta), 1)) ** 2 * np.reshape(dvar, (len(dvar), 1))) * v[1]
    gradnll[2] = -(1 / 2) * np.sum(np.reshape(dzeta, (len(dzeta), 1)) ** 2 * np.reshape(dvar, (len(dvar), 1))) * v[2]

    # Gradient wrt mu
    p = np.fft.fft(v[1] * dvar * zeta - reswt) - 1j * v[2] * w * np.fft.fft(dvar * dzeta)
    gradnll[4:(n + 4)] = - np.sum(a.H * np.real(np.fft.ifft(exp_iweta * p)), axis=1)

    # Gradient wrt logA
    term = (vtot - valpha) * dvar - reswt * zeta
    gradnll[(4 + n):(4 + n + m)] = -np.conj(np.sum(term, axis=0)).T

    # Gradient wrt eta
    ddzeta = np.real(np.fft.ifft(-np.tile(w, m) ** 2 * zeta_f))
    gradnll[(4 + n + m):(4 + n + 2 * m)] = np.sum(dvar * (zeta * dzeta * v[1] + dzeta * ddzeta * v[2]) - reswt * dzeta)

    return [loglik, gradnll]
