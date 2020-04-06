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

% Define transfer function, ideal output pulse
tfun = @(theta,w) theta(1)*exp(1i*theta(2)*w*T);

A0 = 0.25;          % amplitude ratio between pulses
eta0 = 1;           % delay between pulses [T]
theta0 = [A0;eta0]; % Initial parameter vector
Np = length(theta0);

h = tdtf(tfun, theta0, N, T);
g = eye(N)/h;

psi = h*mu;

% Run Monte Carlo
Nmc = pow2(16);
rng('default')

sigbeta = (0:2:20)*1e-2;
Nsig = length(sigbeta);
EQ = zeros(Nsig, 1);
VQ = zeros(Nsig, 1);
EQappx = zeros(Nsig, 1);
EQappx2 = zeros(Nsig, 1);

for i = 1:Nsig
    sigma = [1e-4, sigbeta(i), 0];
    Q = zeros(Nmc,1);
    sigmu = noiseamp(sigma, mu, T);
    sigpsi = noiseamp(sigma, psi, T);
    Vmu = diag(sigmu.^2);
    Vpsi = diag(sigpsi.^2);
    Vt = Vmu + g*Vpsi*g';
    Vti = eye(N)/Vt;
    parfor k = 1:Nmc
        x = mu + sigmu.*randn(N, 1);
        y = psi + sigpsi.*randn(N, 1);
        Vx = diag(noisevar(sigma, x, T));
        Vy = diag(noisevar(sigma, y, T));
        Q(k) = (x - g*y)'*(eye(N)/(Vx + g*Vy*g'))*(x - g*y);
    end
    EQ(i) = mean(Q);
    VQ(i) = var(Q);
    EQappx(i) = N - sigma(2)^2*(3*sum(diag(Vmu).^2.*diag(Vti).^2) ...
        + 2*sum(diag(Vmu).*diag(Vpsi).*diag(Vti*g).^2) ...
        + 3*sum(diag(Vpsi).^2.*diag(g'*Vti*g).^2));
end

figure('Name','Expectation versus sigma_beta')
plot(sigbeta, EQ, 'ko', sigbeta, EQappx, '-')
legend('Simulated','Approximate')
xlabel('\sigma_\beta')
ylabel('E(Q)')

% figure('Name','Variance versus sigma_beta')
% plot(sigbeta, VQ, 'ko')
% xlabel('\sigma_\beta')
% ylabel('Var(Q)')


% sigbeta = (0:20)*1e-3;
% Nsig = length(sigbeta);
% C1MCbeta = zeros(Nsig,1);
% C2MCbeta = zeros(Nsig,1);
% C1appxbeta = zeros(Nsig,1);
% C1appxbeta2 = zeros(Nsig,1);
% C2appxbeta = zeros(Nsig,1);
% for i = 1:Nsig
%     sigma = [1e-4 sigbeta(i) 1e-3];
%     Vmu = noisevar(sigma,y,T);
%     sigmamu = noiseamp(sigma,y,T);
%     resnorm = zeros(Nmc,1);
%     for k = 1:Nmc
%         ym = y + sigmamu.*randn(N, 1);    
%         resnorm(k) = sum((ym-y).^2.*1./noisevar(sigma,ym,T));
%     end
%     
%     C1MCbeta(i) = mean(resnorm);
%     C2MCbeta(i) = var(resnorm);
%     
%     D = tdtf(@(theta, w) -1i*w, 0, N, T);
%     ydot = D*y;
%     C1appxbeta(i) = N - 3*N*sigma(2)^2 - sigma(3)^2*(1./Vmu)'*D.^2*Vmu ...
%         + 12*sigma(2)^4*sum(y.^2./Vmu) ...
%         + 4*sigma(3)^4*((D*y).^2./Vmu.^2)'*D.^2*Vmu ...
%         - sigma(3)^4*(1./Vmu)'*D.^2*Vmu;
%     C1appxbeta2(i) = N - 3*N*sigma(2)^2 - sigma(3)^2*(1./Vmu)'*D.^2*Vmu;
%     C2appxbeta(i) = 2*(2*N - C1appxbeta(i));
% 
% end

% figure('Name','Expectation versus sigma_beta')
% plot(sigbeta,C1MCbeta,'ko', sigbeta, C1appxbeta, '-', ...
%     sigbeta, C1appxbeta2, '-')
% legend('MC','Fourth order','Second order')
% xlabel('\sigma_\beta')
% ylabel('E(C)')
% 
% figure('Name','Simulated variance versus sigma_beta')
% plot(sigbeta,C2MCbeta,'ko',sigbeta, C2appxbeta, '-')
% xlabel('\sigma_\beta')
% ylabel('Var(C)')

% sigtau = (0:20)*1e-4;
% Nsig = length(sigtau);
% C1MCtau = zeros(Nsig,1);
% C2MCtau = zeros(Nsig,1);
% C1appxtau = zeros(Nsig,1);
% C1appxtau2 = zeros(Nsig,1);
% C2appxtau = zeros(Nsig,1);
% for i = 1:Nsig
%     sigma = [1e-4 1e-2 sigtau(i)];
%     Vmu = noisevar(sigma,y,T);
%     sigmamu = noiseamp(sigma,y,T);
%     resnorm = zeros(Nmc,1);
%     for k = 1:Nmc
%         ym = y + sigmamu.*randn(N, 1);    
%         resnorm(k) = sum((ym-y).^2.*1./noisevar(sigma,ym,T));
%     end
%     
%     C1MCtau(i) = mean(resnorm);
%     C2MCtau(i) = var(resnorm);
%     
%     D = tdtf(@(theta, w) -1i*w, 0, N, T);
%     ydot = D*y;
%     C1appxtau(i) = N - 3*N*sigma(2)^2 - sigma(3)^2*(1./Vmu)'*D.^2*Vmu ...
%         + 12*sigma(2)^4*sum(y.^2./Vmu) ...
%         + 4*sigma(3)^4*((D*y).^2./Vmu.^2)'*D.^2*Vmu ...
%         - sigma(3)^4*(1./Vmu)'*D.^2*Vmu;
%     C1appxtau2(i) = N - 3*N*sigma(2)^2 - sigma(3)^2*(1./Vmu)'*D.^2*Vmu;
%     C2appxtau(i) = 2*(2*N - C1appxtau(i));
% 
% end

% figure('Name','Simulated expectation versus sigma_tau')
% plot(sigtau,C1MCtau,'ko', sigtau, C1appxtau, '-', sigtau, C1appxtau2, '-')
% legend('MC','Fourth order','Second order')
% xlabel('\sigma_\tau')
% ylabel('E(C)')
% 
% figure('Name','Simulated variance versus sigma_tau')
% plot(sigtau,C2MCtau,'ko', sigtau, C2appxtau, '-')
% xlabel('\sigma_\tau')
% ylabel('Var(C)')

% Stop timer
toc(tStart)