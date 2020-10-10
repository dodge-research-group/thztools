%% Monte Carlo simulation of time-domain fits
%

%% Start timer
tStart = tic;

warning('off','all')
%% Set initial parameters

T = .05;            % sampling time [ps]
N = 256;            % number of sampled points
Mvec = (10:10:100); % number of traces to compare
Mnum = length(Mvec);
SNR = 2e3;          % signal to noise ratio
w = 0.2;            % pulse width [ps]
tc = N*T/3;         % pulse center [ps]
nMC = pow2(10);      % number of Monte Carlo runs

% Generate ideal time-domain pulse and pulse derivative
t=T*(0:N-1);
t=t(:);

xfun = @(t,t0,w) (1-2*((t-t0)/w).^2).*exp(-((t-t0)/w).^2);

% Compute derivative matrix
fun = @(theta,w) -1i*w;
D = tdtf(fun,0,N,T);

%% Run Monte Carlo simulation

Noise.add = 1/SNR;
Noise.mult = 10/SNR;
Noise.time = 10*T/SNR;
sigma_alpha = Noise.add;
sigma_beta = Noise.mult;
sigma_tau = Noise.time;
Init = cell(nMC,1);

vMean = zeros(3, Mnum);
vStdErr = zeros(3, Mnum);

rng('default')

for iM = 1:Mnum
    M = Mvec(iM);
    vEst = zeros(3,nMC);
    
    fprintf('Iteration: %d/%d\n',iM,Mnum)
        
    parfor jMC=1:nMC
        % Generate noisy data
        x = zeros(N,M);
        for k = 1:M
            x(:,k)=xfun(t,tc+sigma_tau*randn(N,1),w);
        end
        Xn = x + sigma_alpha*randn(N,M) + sigma_beta*x.*randn(N,M);
        
        Fix = struct('logv',false,'mu',true,'A',true,'eta',true);
        Ignore = struct('A',true,'eta',true);
        v0 = mean(var(Xn,1,2))*[1;1;1];
        mu0 = mean(Xn,2);
        Options = struct('v0',v0,'mu0',mu0,'ts',T,'Fix',Fix,'Ignore',Ignore);
        P0 = tdnoisefit(Xn,Options);
        v0 = P0.var;
        
        Fix = struct('logv',false,'mu',false,'A',false,'eta',false);
        Ignore = struct('A',false,'eta',false);
        Options = struct('v0',v0,'mu0',mu0, ...
            'A0',ones(M,1),'eta0',zeros(M,1),'ts',T,'Fix',Fix,'Ignore',Ignore);
        P = tdnoisefit(Xn,Options);
        
        vEst(:,jMC) = P.var;
        
    end
    
    vMean(:,iM) = mean(vEst, 2);
    vStdErr(:,iM) = std(vEst, 0, 2)/sqrt(nMC);
end

%% Stop timer
toc(tStart)

%% Save results
strNow = char(datetime('now','Format','yyyy-MM-dd''T''HHmmss'));
save(['biasfit-' strNow], 'Mvec', 'vMean', 'vStdErr')
disp(strNow)