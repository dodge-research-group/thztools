import numpy as np

from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.tdtf import tdtf


def tdnll(x, param, varargin):
    # Parse function inputs
    [N, M] = x.shape
    # validateattributes(x, {'double'}, {'2d'})
    # validateattributes(Param, {'struct'}, {'nonempty'})
    if nargin > 2:
        Fix = varargin{1}
        # validateattributes(Fix, {'struct'}, {'nonempty'})
    else:
        Fix = {'logv': False, 'mu': False, 'A': False, 'eta': False}

    # Parse parameter dictionary
    Pfields = param.keys()
    Ignore = dict()
    if 'logv' in Pfields:
        v = np.exp(param.get('logv'))
        v = np.reshape(v, (len(v), 1))
        # validateattributes(v, {'double'}, {'vector', 'numel', 3})
    else:
        # error('TDNLL requires Param structure with logv field')
    if 'mu' in Pfields:
        mu = param.get('mu')
        mu = np.reshape(mu, (len(mu), 1))
        # validateattributes(mu, {'double'}, {'vector', 'numel', N})
    else:
        # error('TDNLL requires Param structure with mu field')
    if 'A' in Pfields and param.get('A') is not None:
        A = param.get('A')
        A = np.reshape(A, (len(A), 1))
        # validateattributes(A, {'double'}, {'vector', 'numel', M})
        Ignore['A'] = False
    else:
        A = np.ones((M, 1))
        Ignore['A'] = True
    if 'eta' in Pfields and param.get('eta') is not None:
        eta = param.get('eta')
        eta = np.reshape(eta, (len(eta), 1))
        # validateattributes(eta, {'double'}, {'vector', 'numel', M})
        Ignore['eta'] = False
    else:
        eta = np.zeros((M, 1))
        Ignore['eta'] = True
    if 'ts' in Pfields:
        ts = param.get('ts')
        # validateattributes(ts, {'double'}, {'scalar'})
    else:
        ts = 1
        # warning('TDNLL received Param structure without ts field; set to one')
    if 'D' in Pfields:
        D = param.get('D')
        # validateattributes(D, {'double'}, {'size', [N N]})
    else:
        # Compute derivative matrix
        fun = lambda theta, w: - 1j * w
        D = tdtf(fun, 0, N, ts)

    # Compute frequency vector and Fourier coefficients of mu
    f = fftfreq(N, ts)
    w = 2 * np.pi * f
    mu_f = np.fft.fft(mu)

    gradcalc = ~[Fix.logv; Fix.mu; (Fix.A | | Ignore.A); (Fix.eta | | Ignore.eta)]

    if Ignore.get('eta'):
        zeta = mu * np.conj(A).T
        zeta_f = fft(zeta)
    else:
        exp_iweta = exp(1j * w(:, ones(1, M)).*eta(:, ones(1, N))');
        zeta_f = A(:, ones(1, N))'.*conj(exp_iweta).*mu_f(:,ones(1,M));
        zeta = real(ifft(zeta_f));

        # Compute negative - log likelihood and gradient

        # Compute residuals and their squares
        for subsequent computations
        res = x - zeta;
        ressq = res.^ 2;

        # Simplest case: just variance and signal parameters, A and eta fixed at
        # defaults
        if Ignore.A & & Ignore.eta

        Dmu = real(ifft(1
        i * w. * mu_f));
        valpha = v(1);
        vbeta = v(2) * mu. ^ 2;
        vtau = v(3) * Dmu. ^ 2;
        vtot = valpha + vbeta + vtau;

        resnormsq = ressq. / vtot(:, ones(1, M));
        nll = M * N * log(2 * pi) / 2 + (M / 2) * sum(log(vtot)) + sum(resnormsq(:)) / 2;

        # Compute gradient if requested
        if nargout > 1
        Ngrad = sum(gradcalc(1:2).*[3;N]);
        gradnll = zeros(Ngrad, 1);
        nStart = 1;
        dvar = (vtot - mean(ressq, 2)). / vtot. ^ 2;
        if gradcalc(1)
        gradnll(nStart) = (M / 2) * sum(dvar) * v(1);
        gradnll(nStart+1) = (M / 2) * sum(mu.^ 2. * dvar) * v(2);
        gradnll(nStart+2) = (M / 2) * sum(Dmu.^ 2. * dvar) * v(3);
        nStart = nStart + 3;

        if gradcalc(2)
        gradnll(nStart+(0:N - 1)) = M * (v(2) * mu. * dvar...
                                         + v(3) * transpose(D) * (Dmu. * dvar)...
                                         - mean(res, 2). / vtot);

        # Alternative case: A, eta, or both are not set to defaults
        else

        Dzeta = real(ifft(1
        i * w(:, ones(1, M)).*zeta_f));

        valpha = v(1);
        vbeta = v(2) * zeta. ^ 2;
        vtau = v(3) * (Dzeta). ^ 2;
        vtot = valpha + vbeta + vtau;

        resnormsq = ressq. / vtot;
        nll = M * N * log(2 * pi) / 2 + sum(log(vtot(:))) / 2 + sum(resnormsq(:)) / 2;

        # Compute gradient if requested
        if nargout > 1
        Ngrad = sum(gradcalc.*[3;N;M;M]);
        gradnll = zeros(Ngrad, 1);
        nStart = 1;
        reswt = res./ vtot;
        dvar = (vtot - ressq)./ vtot.^ 2;
        if gradcalc(1)
        % Gradient wrt logv
        gradnll(nStart) = (1 / 2) * sum(dvar(:))*v(1);
        gradnll(nStart + 1) = (1 / 2) * sum(zeta(:).^ 2. * dvar(:))*v(2);
        gradnll(nStart + 2) = (1 / 2) * sum(Dzeta(:).^ 2. * dvar(:))*v(3);
        nStart = nStart + 3;

        if gradcalc(2)
           # Gradient wrt mu
        P = fft(v(2) * dvar. * zeta - reswt)...
            - 1
        i * v(3) * w. * fft(dvar. * Dzeta);
        gradnll(nStart: nStart + N - 1) = ...
        sum(A
        '.*real(ifft(exp_iweta.*P)),2);
        nStart = nStart + N;
        end
        if gradcalc(3)
           % Gradient
        wrt
        A
        term = ((vtot - valpha). * dvar - reswt. * zeta);
        gradnll(nStart + (0:M-1)) = sum(term, 1)
        './A;
        nStart = nStart + M;
        end
        if gradcalc(4)
           % Gradient
        wrt
        eta
        DDzeta = ...
        real(ifft(-w(:, ones(1, M)).^ 2. * zeta_f));
        gradnll(nStart + (0:M-1)) = ...
                                    - sum(dvar. * (zeta. * Dzeta * v(2) + Dzeta. * DDzeta * v(3))...
                                          - reswt. * Dzeta);
    return [nll, gradnll]