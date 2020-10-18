%% Monte Carlo simulation of time-domain fits
%

%% Start timer
tStart = tic;

warning('off','all')
%% Set initial parameters

T=.05;          % sampling time [ps]
N=256;          % number of sampled points
M=10;           % number of traces to compare
SNR=2e3;        % signal to noise ratio
w=0.2;          % pulse width [ps]
tc=N*T/3;       % pulse center [ps]
nMC = pow2(10); % number of Monte Carlo runs

% Generate ideal time-domain pulse and pulse derivative
t=T*(0:N-1);
t=t(:);

xfun = @(t,t0,w) (1-2*((t-t0)/w).^2).*exp(-((t-t0)/w).^2);

% % Compute derivative matrix
% fun = @(theta,w) -1i*w;
% D = tdtf(fun,0,N,T);

%% Run Monte Carlo simulation

Noise.add = 1/SNR;
Noise.mult = 10/SNR;
Noise.time = 10*T/SNR;
sigma_alpha = Noise.add;
sigma_beta = Noise.mult;
sigma_tau = Noise.time;
Init = cell(nMC,1);

sigma_A = 0.05;
sigma_eta = 0.1*T;

v0Err = zeros(3,nMC);

muEst = zeros(N,nMC);
vEst = zeros(3,nMC);
AEst = zeros(M,nMC);
etaEst = zeros(M,nMC);
% muErr = zeros(N,nMC);
vErr = zeros(3,nMC);
% AErr = zeros(M-1,nMC);
% etaErr = zeros(M-1,nMC);
nll = zeros(nMC,1);
Diagnostic = struct('exitflag',Init,...
    'output',Init,'grad',Init,'hessian',Init,'Err',Init);
% hks = false(nMC,1);
% pks = zeros(nMC,1);
% ksstat = zeros(nMC,1);
% cvks = zeros(nMC,1);
% 
% had = false(nMC,1);
% pad = zeros(nMC,1);
% adstat = zeros(nMC,1);
% cvad = zeros(nMC,1);
% 
% hrun = false(nMC,1);
% prun = zeros(nMC,1);
% Runstat = struct('nruns',Init,...
%     'n1',Init,...
%     'n0',Init,...
%     'z',Init);

Xn = cell(nMC,1);

A = ones(nMC,M);
eta = zeros(nMC,M);

for iMC=1:nMC
    rng(iMC)
    if ~mod(iMC,round(nMC/10))
        fprintf('Monte Carlo run: %d/%d\n',iMC,nMC)
    end
    
    % Generate noisy data
    x = zeros(N,M);
    A(iMC, 2:M) = 1 + sigma_A*randn(1,M-1);
    eta(iMC, 2:M) = sigma_eta*randn(1,M-1);
    for jj = 1:M
        x(:,jj) = A(iMC,jj)...
            *xfun(t, tc + sigma_tau*randn(N,1) + eta(iMC,jj), w);
    end
    Xn{iMC} = x + sigma_alpha*randn(N,M) + sigma_beta*x.*randn(N,M);
    
    % Fit Xn for A and eta, holding logv constant and using known mu
    Fix = struct('logv', true, ...
        'mu', true, ...
        'A', false, ...
        'eta', false);
    Ignore = struct('A', false, ...
        'eta', false);
    v0 = mean(var(Xn{iMC},1,2))*[1;eps;eps];
    mu0 = xfun(t, tc, w);
    Options = struct('v0', v0, ...
        'mu0', mu0, ...
        'ts', T, ...
        'Fix', Fix, ...
        'Ignore', Ignore);
    P0 = tdnoisefit(Xn{iMC}, Options);

    % Fit for v, eta, and A
    Fix = struct('logv', false, ...
        'mu', true, ...
        'A', true, ...
        'eta', true);
    Ignore = struct('A', false, ...
        'eta', false);
    Options = struct('v0', v0(1)*[1;1;1], ...
        'mu0', mu0, ...
        'A0', P0.A, ...
        'eta0', P0.eta, ...
        'ts', T, ...
        'Fix', Fix, ...
        'Ignore', Ignore);
    [P,nll(iMC),Diagnostic(iMC)] = tdnoisefit(Xn{iMC},Options);
    
    vEst(:,iMC) = P.var;
    muEst(:,iMC) = P.mu;
    AEst(:,iMC) = P.A;
    etaEst(:,iMC) = P.eta;
    
    vErr(:,iMC) = Diagnostic(iMC).Err.var;
%     muErr(:,iMC) = Diagnostic(iMC).Err.mu;
%     AErr(:,iMC) = Diagnostic(iMC).Err.A;
%     etaErr(:,iMC) = Diagnostic(iMC).Err.eta;
% 
%     zeta = zeros(N,M);
%     S = zeros(N,N,M);
%     for m = 1:M
%         S(:,:,m) = shiftmtx(P.eta(m),N,T);
%         zeta(:,m) = P.A(m)*S(:,:,m)*P.mu;
%     end
%     
%     Dmu = D*P.mu;
%     valpha = P.var(1);
%     vbeta = P.var(2)*P.mu.^2;
%     vtau = P.var(3)*(Dmu).^2;
%     vtot = valpha + vbeta + vtau;
% 
%     delta = (Xn{iMC} - zeta)./sqrt(vtot);
%     
%     [hks(iMC),pks(iMC),ksstat(iMC),cvks(iMC)] = kstest(delta(:));
%     [had(iMC),pad(iMC),adstat(iMC),cvad(iMC)] = adtest(delta(:));
%     [hrun(iMC),prun(iMC),Runstat(iMC)] = runstest(delta(:));
%     
end


%% Plot parameter distributions

% figure('Name','sigma_alpha^2');
% histogram(vEst(1,:));
% xlabel('\sigma_\alpha^2')
% 
fprintf('True value: %.4g\n',Noise.add^2)
fprintf('Mean sigma_alpha^2: %.4g\n', mean(vEst(1,:)));
fprintf('Relative bias: %.4f\n', mean(vEst(1,:))/Noise.add^2);
% fprintf('Standard deviation sigma_alpha^2: %.4g\n',std(v0Est(1,:)));
% fprintf('Mean standard error sigma_alpha^2: %.4g\n\n',mean(v0Err(1,:)));
fprintf('\n')

% figure('Name','sigma_beta^2');
% histogram(vEst(2,:));
% xlabel('\sigma_\beta^2')
% 
fprintf('True value: %.4g\n',Noise.mult^2)
fprintf('Mean sigma_beta^2: %.4g\n', mean(vEst(2,:)));
fprintf('Relative bias: %.4f\n', mean(vEst(2,:))/Noise.mult^2);
% fprintf('Standard deviation sigma_beta^2: %.4g\n',std(v0Est(2,:)));
% fprintf('Mean standard error sigma_beta^2: %.4g\n\n',mean(v0Err(2,:)));
fprintf('\n')

% figure('Name','sigma_tau^2');
% histogram(vEst(3,:));
% xlabel('\sigma_\tau^2')
% 
fprintf('True value: %.4g\n',Noise.time^2)
fprintf('Mean sigma_tau^2: %.4g\n', mean(vEst(3,:)));
fprintf('Relative bias: %.4f\n', mean(vEst(3,:))/Noise.time^2);
% fprintf('Standard deviation sigma_tau^2: %.4g\n',std(v0Est(3,:)));
% fprintf('Mean standard error sigma_tau^2: %.4g\n\n',mean(v0Err(3,:)));
fprintf('\n')

% figure('Name','Kolmogorov-Smirnov statistic');
% histogram(ksstat)
% xlabel('D')
% 
% fprintf(...
%     'Number of Kolmogorov-Smirnov rejections (alpha = 5%%): %d/%d\n',...
%     sum(hks),nMC);
% 
% figure('Name','Anderson-Darling statistic');
% histogram(adstat)
% xlabel('A^2')
% 
% fprintf('Number of Anderson-Darling rejections (alpha = 5%%): %d/%d\n',...
%     sum(had),nMC);
% 
% figure('Name','Runs statistic');
% histogram([Runstat.z])
% xlabel('Z')
% 
% fprintf('Number of run test rejections (alpha = 5%%): %d/%d\n',...
%     sum(hrun),nMC);

%% Stop timer
toc(tStart)

%% Save results
strNow = char(datetime('now','Format','yyyy-MM-dd''T''HHmmss'));
% save(['mctdnoise-' strNow])
disp(strNow)