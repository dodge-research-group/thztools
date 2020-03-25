%% Compare approximate cost function distribution moments with simulation
tStart = tic;
N = 256;
T = 0.05;
t0 = 2.5;
[y, t]=thzgen(N, T, t0, 'taur', 0.4);
Nmc = pow2(20);
rng('default')

sigbeta = (0:20)*1e-3;
Nsig = length(sigbeta);
C1MCbeta = zeros(Nsig,1);
C1appxbeta = zeros(Nsig,1);
for i = 1:Nsig
    sigma = [1e-4 sigbeta(i) 1e-3];
    Vmu = noisevar(sigma,y,T);
    sigmamu = noiseamp(sigma,y,T);
    ym = repmat(y, 1, Nmc) + repmat(sigmamu, 1, Nmc).*randn(N, Nmc);
    resnorm = zeros(Nmc,1);
    for k = 1:Nmc
        resnorm(k) = sum((ym(:,k)-y).^2.*1./noisevar(sigma,ym(:,k),T));
    end
    
    C1MCbeta(i) = mean(resnorm);
    
    D = tdtf(@(theta, w) -1i*w, 0, N, T);
    ydot = D*y;
    C1appxbeta(i) = N - 9*N*sigma(2)^2 - 0.5*sigma(3)^2*(1./Vmu)'*D.^2*Vmu;

end

figure('Name','Expectation versus sigma_beta')
plot(sigbeta,C1MCbeta,'ko', sigbeta, C1appxbeta, '-')
xlabel('\sigma_\beta')
ylabel('E(C)')

sigtau = (0:20)*1e-4;
Nsig = length(sigtau);
C1MCtau = zeros(Nsig,1);
C1appxtau = zeros(Nsig,1);
for i = 1:Nsig
    sigma = [1e-4 1e-2 sigtau(i)];
    Vmu = noisevar(sigma,y,T);
    sigmamu = noiseamp(sigma,y,T);
    ym = repmat(y, 1, Nmc) + repmat(sigmamu, 1, Nmc).*randn(N, Nmc);
    resnorm = zeros(Nmc,1);
    for k = 1:Nmc
        resnorm(k) = sum((ym(:,k)-y).^2.*1./noisevar(sigma,ym(:,k),T));
    end
    
    C1MCtau(i) = mean(resnorm);
    
    D = tdtf(@(theta, w) -1i*w, 0, N, T);
    ydot = D*y;
    C1appxtau(i) = N - 9*N*sigma(2)^2 - 0.5*sigma(3)^2*(1./Vmu)'*D.^2*Vmu;
end

figure('Name','Simulated expectation versus sigma_tau')
plot(sigtau,C1MCtau,'ko', sigtau, C1appxtau, '-')
xlabel('\sigma_\tau')
ylabel('E(C)')
toc(tStart)