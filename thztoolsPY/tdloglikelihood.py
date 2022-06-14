import numpy as np

from thztoolsPY.fftfreq import fftfreq


def tdloglikelihood(param, x, ts):
    """
    TDLOGLIKELIHOOD computes log-likelihood for the time-domain noise model

    Syntax:   loglik = tdloglikelihood(Param, x, ts)

    Description:
    TDLOGLIKELIHOOD computes the log-likelihood function for obtaining the
    data matrix x, given the parameter vector Param.

    Parameters
    ----------
    x               Data matrix                             [NxM double]
    param           Parameter vector, including:
                        logv    Log of noise parameters	    [3x1 double]
                        mu      Signal vector               [Nx1 double]
                        logA    Log of amplitude vector     [Mx1 double]
                        eta     Delay vector                [Mx1 double]
    ts              Sampling time                           [1x1 double]

    Returns
    -------
    loglik          log-likelihood function
    gradll          Gradient of the log-likelihood function

    """
    # Parse function inputs
    [N, M] = x.shape

    logv = param[:3]
    v = np.exp(logv)
    mu = param[3:(3 + N)]
    logA = param[(3 + N):(3 + N + M)]
    A = np.exp(logA)
    eta = param[(3 + N + M):(3 + N + 2*M)]

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(N, ts)
    w = 2 * np.pi * f
    mu_f = np.fft.fft(mu)

    # Compute zeta
    exp_iweta = np.exp(1j * np.tile(w, M) * np.conj(np.tile(eta, N)).T)
    zeta_f = np.conj(np.tile(A, N)).T * np.conj(exp_iweta) * np.tile(mu_f, M)
    zeta = np.real(np.fft.ifft(zeta_f))

    # Compute log - likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    Dzeta = np.real(np.fft.ifft(1j * np.tile(w, M) * zeta_f))

    valpha = v[0]
    vbeta = v[1] * zeta**2
    vtau = v[2] * Dzeta**2
    vtot = valpha + vbeta + vtau

    resnormsq = ressq / vtot
    loglik = -M * N * np.log(2 * np.pi) / 2 - np.sum(np.log(vtot)) / 2 - np.sum(resnormsq) / 2

    # Compute gradient
    Ngrad = 3 + N + 2 * M
    gradll = np.zeros((Ngrad, 1))

    reswt = res / vtot
    dvar = (vtot - ressq) / vtot**2

    # Gradient wrt logv
    gradll[0] = -(1 / 2) * np.sum(dvar) * v[0]
    gradll[1] = -(1 / 2) * np.sum(np.reshape(zeta, (len(zeta), 1))**2 * np.reshape(dvar, (len(dvar), 1))) * v[1]
    gradll[2] = -(1 / 2) * np.sum(np.reshape(Dzeta, (len(Dzeta), 1))**2 * np.reshape(dvar, (len(dvar), 1))) * v[2]

    # Gradient wrt mu
    P = np.fft.fft(v[1] * dvar * zeta - reswt) - 1j * v[2] * w * np.fft.fft(dvar * Dzeta)
    gradll[4:(N + 4)] = - np.sum(A.H * np.real(np.fft.ifft(exp_iweta*P)), axis=1)

    # Gradient wrt logA
    term = (vtot - valpha) * dvar - reswt * zeta
    gradll[(4 + N):(4 + N + M)] = -np.conj(np.sum(term, axis=0)).T

    # Gradient wrt eta
    DDzeta = np.real(np.fft.ifft(-np.tile(w, M)**2 * zeta_f))
    gradll[(4 + N + M):(4 + N + 2*M)] = np.sum(dvar * (zeta * Dzeta * v[1] + Dzeta * DDzeta * v[2]) - reswt * Dzeta)

    return [loglik, gradll]
