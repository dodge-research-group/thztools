%% Monte Carlo simulation of time-domain fit
%
tic
%% Set initial parameters

T=.05;              % sampling time [ps]
N=256;              % number of sampled points
sigma_alpha=1e-3;   % amplitude noise [units of time-domain peak]
sigma_beta=1e-2;    % multiplicative noise [dimensionless]
sigma_tau=1e-3;     % time base noise [ps]
w=0.2;              % pulse width [ps]
t0=N*T/3;           % pulse center [ps]
alpha = 0;          % offset of the refrence pulse
beta = .004;        % offset of the sample pulse
nMC = pow2(10);

%% Generate time array and two ideal pulses, y1 and y2

[y1,t]=pulsegen(N,t0,w,1,T);


ParmIn.eta=2;      % delay between pulses [T]
ParmIn.a = 1;
ParmIn.b = 0.25;      % amplitude ratio between pulses

y2 = pulserationaldelay(y1,ParmIn) ;


TDFit.x0 = [ParmIn.b(1);ParmIn.eta;0];
TDFit.lb = [];
TDFit.ub = [];
TDFit.solver = 'lsqnonlin';
TDFit.options = optimoptions('lsqnonlin',...
    'Display','Off');

tfun = @(theta,w) theta(1)*exp(1i*theta(2)*w*T);

Noise = struct('add',sigma_alpha,...
    'mult',sigma_beta,...
    'time',sigma_tau);
nNoise = length(Noise);

%%
% Run Monte Carlo simulations with each noise model.

Init = cell(nMC,nNoise);
ParmOutFDnoise = struct('a',Init,'b',Init,'eta',Init);
Fx = ParmIn;
Fx.b = NaN;
Fx.eta = NaN;
Np = sum(isnan([Fx.a;Fx.b;Fx.eta]))+1;
resnormFDnoise = zeros(nMC,nNoise);
DiagnosticFDnoise = struct('exitflag',Init,...
    'output',Init,...
    'jacobian',Init,...
    'covariance',Init,...
    'chisqfn',Init,...
    'noise',Init,...
    'init',Init,...
    'freq',Init);

ParmOutTDnoise = zeros(nMC,nNoise,Np);
resnormTDnoise = zeros(nMC,nNoise);
DiagnosticTDnoise = struct('exitflag',Init,...
    'jacobian',Init);


for iNoise = 1:nNoise
    
    Vx = diag(Noise(iNoise).add^2 + (Noise(iNoise).mult*y1).^2 + ...
        (Noise(iNoise).time*(gradient(y1,t))).^2);
    Vy = diag(Noise(iNoise).add^2 + (Noise(iNoise).mult*y2).^2 + ...
        (Noise(iNoise).time*(gradient(y2,t))).^2);

    for jMC=1:nMC
        
        % Generate noisy data
        yn1 = y1 + ...
            Noise(iNoise).add*randn(N,1) + ...
            Noise(iNoise).mult*y1.*randn(N,1) + ...
            Noise(iNoise).time*(gradient(y1,t)).*randn(N,1) + alpha;
        yn2 = y2 +  ...
            Noise(iNoise).add*randn(N,1) + ...
            Noise(iNoise).mult*y2.*randn(N,1) + ...
            Noise(iNoise).time*(gradient(y2,t)).*randn(N,1) + beta;
        
        TDFit.objective = @(theta) ...
            costfunwofflsq(tfun,theta(1:end-1),yn1,yn2,...
            alpha,theta(end),Vx,Vy,T);
        [ParmOutTDnoise(jMC,iNoise,:),...
            resnormTDnoise(jMC,iNoise),~,...
            DiagnosticTDnoise(jMC,iNoise).exitflag,~,~,...
            DiagnosticTDnoise(jMC,iNoise).jacobian] = lsqnonlin(TDFit);
        [Q,R]=qr(DiagnosticTDnoise(jMC,iNoise).jacobian);
        DiagnosticTDnoise(jMC,iNoise).covariance = ...
            full(eye(size(R,2))/(R'*R));

    end
    
    cvTDnoise = ...
        mean(reshape([DiagnosticTDnoise(:,iNoise).covariance],Np,Np,nMC),3);

    figure('Name',['Delay histogram, time-domain fit, noise model'...
        int2str(iNoise)]);
    histogram(ParmOutTDnoise(:,iNoise,2)*T*1000)
    xlabel('\eta (fs)')
    ylabel('Occurence')
    title(sprintf('TD Noise model %d',iNoise))
    
    figure('Name',['Delay normal probability plot, ',...
        'time-domain fit, noise model' int2str(iNoise)]);
    normplot(ParmOutTDnoise(:,iNoise,2)*T*1000)
    xlabel('\eta (fs)')
    title(sprintf('TD Noise model %d',iNoise))
    
    figure('Name',['Amplitude histogram, time-domain fit, ',...
        'noise model' int2str(iNoise)]);
    histogram(ParmOutTDnoise(:,iNoise,1))
    xlabel('b_0')
    ylabel('Occurence')
    title(sprintf('TD Noise model %d',iNoise))
    
    figure('Name',['Amplitude normal probability plot, ',...
        'time-domain fit, noise model' int2str(iNoise)]);
    normplot(ParmOutTDnoise(:,iNoise,1))
    xlabel('b_0')
    title(sprintf('TD Noise model %d',iNoise))
    
    figure('Name',['Cost-function histogram, time-domain fit, ',...
        'noise model' int2str(iNoise)]);
    histogram(resnormTDnoise(:,iNoise))
    xlabel('Norm of residuals')
    ylabel('Occurence')
    title(sprintf('TD Noise model %d',iNoise))

    figure('Name',['Cost-function cumulative distribution, ',...
        'time-domain fit, noise model ' int2str(iNoise)]);
    ecdf(resnormTDnoise(:,iNoise),'bounds','on');
    hold on
    r = floor(min(resnormTDnoise(:,iNoise))):...
        ceil(max(resnormTDnoise(:,iNoise)));
    nu = N-Np;
    p = chi2cdf(r,nu);
    plot(r,p,'k-')
    legend('Empirical','LCB','UCB',...
        '\chi^2(r | N_t - N_p)',...
        'Location','SE')
    xlabel('Norm of residuals')
    hold off
    title(sprintf('TD Noise model %d', iNoise))
    
    fprintf('***** NOISE MODEL %d *****\n',iNoise)
    fprintf('nu:\t\t%.4g\n',nu)
    fprintf('mean(resnormTD):\t%.4g\n\n',mean(resnormTDnoise(:,iNoise)))
    
    fprintf('mean(b0TD):\t\t%#.4g\n',mean(ParmOutTDnoise(:,iNoise,1)))
    fprintf('std(b0TD), actual:\t%#.2g\n',...
        std(ParmOutTDnoise(:,iNoise,1)))
    fprintf('std(b0TD), mean estimate:\t%#.2g\n\n',sqrt(cvTDnoise(1,1)))
    
    fprintf('mean(etaTD*T):\t\t\t%#.4g\n',...
        mean(ParmOutTDnoise(:,iNoise,2))*T)
    fprintf('std(etaTD*T), actual:\t\t%#.2g\n',...
        std(ParmOutTDnoise(:,iNoise,2))*T)
    fprintf('std(etaTD*T), mean estimate:\t%#.2g\n\n',...
        sqrt(cvTDnoise(2,2))*T)
    
end


%%
toc