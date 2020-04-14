%% Compare approximate cost function distribution moments with simulation

% Start timer
tStart = tic;

% Create ideal input pulse
N = 256;
T = 0.05;
t0 = 2.5;
[mu, t] = thzgen(N, T, t0, 'taur', 0.4);

f = fftfreq(N, T);
w = 2*pi*f;

% Define transfer function
tfun = @(theta,w) theta(1)*exp(1i*theta(2)*w*T);

D = tdtf(@(theta, w) -1i*w, 0, N, T);

% Run Monte Carlo
Nmc = pow2(18);
rng('default')

A = [1, 0.25];
eta = [0, 1];
sigbeta = (0:2:20)*1e-3;
Nbeta = length(sigbeta);
EQbeta = zeros(Nbeta, 1);
VQbeta = zeros(Nbeta, 1);
EQbetaAppx = zeros(Nbeta, 1);

for m = 1:length(A)
    for n = 1:length(eta)
        A0 = A(m);             % amplitude ratio between pulses
        eta0 = eta(n);           % delay between pulses [T]
        theta0 = [A0;eta0]; % Initial parameter vector
        Np = length(theta0);
        
        h = tdtf(tfun, theta0, N, T);
        g = eye(N)/h;
        
        psi = h*mu;
        
        
        for i = 1:Nbeta
            sigma = [1e-4, sigbeta(i), 1e-3];
            Qbeta = zeros(Nmc,1);
            sigmu = noiseamp(sigma, mu, T);
            sigpsi = noiseamp(sigma, psi, T);
            Vmu = diag(sigmu.^2);
            Vpsi = diag(sigpsi.^2);
            Vt = Vmu + g*Vpsi*g';
            Vti = eye(N)/Vt;
            parfor k = 1:Nmc
                xbeta = mu + sigmu.*randn(N, 1);
                ybeta = psi + sigpsi.*randn(N, 1);
                Vxbeta = diag(noisevar(sigma, xbeta, T));
                Vybeta = diag(noisevar(sigma, ybeta, T));
                Qbeta(k) = (xbeta - g*ybeta)'*(eye(N)...
                    /(Vxbeta + g*Vybeta*g'))...
                    *(xbeta - g*ybeta);
            end
            EQbeta(i) = mean(Qbeta);
            VQbeta(i) = var(Qbeta);
            EQbetaAppx(i) = N - sigma(2)^2*(...
                3*sum(diag(Vmu).^2.*diag(Vti).^2) ...
                + 2*sum(diag(Vmu).*diag(Vpsi).*diag(Vti*g).^2) ...
                + 3*sum(diag(Vpsi).^2.*diag(g'*Vti*g).^2)) ...
                - sigma(3)^2*(diag(Vmu)'*(Vti.^2)*(D.^2)*diag(Vmu) ...
                + 2*diag(Vmu)'*(Vti.*(D'))*(Vti.*D)*diag(Vmu) ...
                + diag(Vmu)'*((Vti*g).^2)*(D.^2)*diag(Vpsi) ...
                + diag(Vmu)'*((D').^2)*((Vti*g).^2)*diag(Vpsi) ...
                + diag(Vpsi)'*((g'*Vti*g).^2)*(D.^2)*diag(Vpsi) ...
                + 2*diag(Vpsi)'*((g'*Vti*g).*(D'))*((g'*Vti*g).*D)...
                *diag(Vpsi));
        end
        
        figure('Name','Expectation versus sigma_beta')
        plot(sigbeta, EQbeta, 'ko', sigbeta, EQbetaAppx, '-')
        legend('Simulated','Approximate')
        xlabel('\sigma_\beta')
        ylabel('E(Q)')
        title(sprintf('E(Q), A = %.2f, eta = %.1f',A0,eta0))
        
        figure('Name','Variance versus sigma_beta')
        plot(sigbeta, VQbeta, 'ko', sigbeta, 2*N*ones(size(sigbeta)), '-')
        legend('Simulated','Approximate')
        xlabel('\sigma_\beta')
        ylabel('Var(Q)')
        title(sprintf('Var(Q) A = %.2f, eta = %.1f',A0,eta0))
        
        sigtau = (0:2:20)*1e-4;
        Ntau = length(sigtau);
        EQtau = zeros(Ntau, 1);
        VQtau = zeros(Ntau, 1);
        EQtauAppx = zeros(Ntau, 1);
        
        for i = 1:Ntau
            sigma = [1e-4, 1e-2, sigtau(i)];
            Qtau = zeros(Nmc,1);
            sigmu = noiseamp(sigma, mu, T);
            sigpsi = noiseamp(sigma, psi, T);
            Vmu = diag(sigmu.^2);
            Vpsi = diag(sigpsi.^2);
            Vt = Vmu + g*Vpsi*g';
            Vti = eye(N)/Vt;
            parfor k = 1:Nmc
                xtau = mu + sigmu.*randn(N, 1);
                ytau = psi + sigpsi.*randn(N, 1);
                Vxtau = diag(noisevar(sigma, xtau, T));
                Vytau = diag(noisevar(sigma, ytau, T));
                Qtau(k) = (xtau - g*ytau)'...
                    *(eye(N)/(Vxtau + g*Vytau*g'))...
                    *(xtau - g*ytau);
            end
            EQtau(i) = mean(Qtau);
            VQtau(i) = var(Qtau);
            EQtauAppx(i) = N - sigma(2)^2*(...
                3*sum(diag(Vmu).^2.*diag(Vti).^2) ...
                + 2*sum(diag(Vmu).*diag(Vpsi).*diag(Vti*g).^2) ...
                + 3*sum(diag(Vpsi).^2.*diag(g'*Vti*g).^2)) ...
                - sigma(3)^2*(diag(Vmu)'*(Vti.^2)*(D.^2)*diag(Vmu) ...
                + 2*diag(Vmu)'*(Vti.*(D'))*(Vti.*D)*diag(Vmu) ...
                + diag(Vmu)'*((Vti*g).^2)*(D.^2)*diag(Vpsi) ...
                + diag(Vmu)'*((D').^2)*((Vti*g).^2)*diag(Vpsi) ...
                + diag(Vpsi)'*((g'*Vti*g).^2)*(D.^2)*diag(Vpsi) ...
                + 2*diag(Vpsi)'*((g'*Vti*g).*(D'))*((g'*Vti*g).*D)...
                *diag(Vpsi));
        end
        
        figure('Name','Expectation versus sigma_tau')
        plot(sigtau, EQtau, 'ko', sigtau, EQtauAppx, '-')
        legend('Simulated','Approximate')
        xlabel('\sigma_\tau')
        ylabel('E(Q)')
        title(sprintf('E(Q), A = %.2f, eta = %.1f',A0,eta0))
        
        figure('Name','Variance versus sigma_tau')
        plot(sigtau, VQtau, 'ko', sigtau, 2*N*ones(size(sigtau)), '-')
        legend('Simulated','Approximate')
        xlabel('\sigma_\tau')
        ylabel('V(Q)')
        title(sprintf('Var(Q) A = %.2f, eta = %.1f',A0,eta0))
        
        fprintf("Finished loop: %d, %d\n", m, n)
        toc(tStart)
    end
end