import numpy as np

from thztoolsPY.fftfreq import fftfreq


def tdloglikelihood(param, x, ts):
    # Parse function inputs
    [N, M] = x.shape

    logv = param[:3]
    v = np.exp(logv)
    mu = param[4:(N + 4)]
    logA = param[(4 + N):(4 + N + M)]
    A = np.exp(logA)
    eta = param[(4 + N + M):(4 + N + 2*M)]

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(N, ts)
    w = 2 * np.pi * f
    mu_f = np.fft(mu)

    # Compute zeta
    exp_iweta = np.exp(1j * w(:, ones(1, M)) * eta(:, ones(1, N))')
    zeta_f = A(:, ones(1, N))' *conj(exp_iweta).*mu_f(:,ones(1,M))
    zeta = np.real(np.fft.ifft(zeta_f))

    # Compute log - likelihood and gradient

    # Compute residuals and their squares for subsequent computations
    res = x - zeta
    ressq = res**2

    Dzeta = real(ifft(1i * w(:, ones(1, M)) * zeta_f))

    valpha = v[0]
    vbeta = v[1] * zeta**2
    vtau = v[2] * (Dzeta)**2
    vtot = valpha + vbeta + vtau

    resnormsq = ressq / vtot
    loglik = -M * N * np.log(2 * np.pi) / 2 - np.sum(np.log(vtot(:))) / 2 - np.sum(resnormsq(:)) / 2

    # Compute gradient if requested
    if nargout > 1:
        Ngrad = 3 + N + 2*M
        gradll = np.zeros((Ngrad, 1))

        reswt = res / vtot
        dvar = (vtot - ressq) / vtot**2

        # Gradient wrt logv
        gradll[0] = -(1 / 2) * np.sum(dvar(:))*v[0]
        gradll[1] = -(1 / 2) * np.sum(zeta(:)**2 * dvar(:))*v[1]
        gradll[2] = -(1 / 2) * np.sum(Dzeta(:)**2 * dvar(:))*v[2]

        # Gradient wrt mu
        P = np.fft.fft(v[1] * dvar. * zeta - reswt) - 1j * v[2] * w * np.fft.fft(dvar * Dzeta)
        gradll[4:(N + 4)] = - np.sum(A'.*np.real(np.fft.ifft(exp_iweta.*P)),2)

        # Gradient wrt logA
        term = (vtot - valpha) * dvar - reswt * zeta
        gradll[(4 + N):(4 + N + M)] = -np.sum(term, 1)'

        # Gradient wrt eta
        DDzeta = np.real(np.fft.ifft(-w(:, ones(1, M))**2 * zeta_f))
        gradll[(4 + N + M):(4 + N + 2*M)] = np.sum(dvar * (zeta * Dzeta * v[1] + Dzeta * DDzeta * v[2])- reswt * Dzeta)

    return [loglik, gradll]
