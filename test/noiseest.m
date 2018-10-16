%% Fit noise model to air scan data

%% Set path
curpath = path;
curDir = fileparts(mfilename('fullpath'));
fs = strfind(curDir,filesep);
topDir = fullfile(curDir(1:fs(end)));
libDir = fullfile(curDir(1:fs(end)),'lib');
if ~contains(curpath,libDir)
    addpath(libDir);
end

%% Start timer
tStart = tic;

%% Load data
dataDir = fullfile(curDir(1:fs(end)),...
    'dat','2017-03-20','Air Scans','Normal');
Data = thzload(dataDir);
   
t = [Data.Time];
t = t(:,1);
gain = 5e7;  %gain is 5e7 V/A
X = [Data.Amplitude].*1e12/gain; % in pA

% Determine sampling time
dt = diff(t);
T = mean(dt(:));

% Compute derivative matrix
[N,M] = size(X);
fun = @(theta,w) -1i*w;
D = tdtf(fun,0,N,T);

% Initialize parameter structure
iFit = 1;
P = struct('var',[],'mu',[],'A',[],'eta',[],'ts',[]);

%% Fit for delay
% Assume constant noise, average signal, and constant amplitude

Fix = struct('logv',true,'mu',true,'A',true,'eta',false);
Ignore = struct('A',true,'eta',false);
v0 = mean(var(X,1,2))*[1;eps;eps];
mu0 = mean(X,2);
Options = ...
    struct('v0',v0,'mu0',mu0,'ts',T,'Fix',Fix,'Ignore',Ignore);

P(iFit) = tdnoisefit(X,Options);
eta0 = P(iFit).eta;
iFit = iFit + 1;
    
%% Fit for amplitude
% Assume constant noise, average signal, and delays from previous fit

Fix = struct('logv',true,'mu',true,'A',false,'eta',true);
Ignore = struct('A',false,'eta',true);
Options = struct('v0',v0,'mu0',mu0,'eta0',eta0,'ts',T, ...
    'Fix',Fix,'Ignore',Ignore);

P(iFit) = tdnoisefit(X,Options);
A0 = P(iFit).A;
iFit = iFit + 1;

%% Revise mu0

Xadjusted = airscancorrect(X,P(end));
mu0 = mean(Xadjusted,2);

%% Fit for var
% Assume constant signal, amplitude, and delays from previous fits

Fix = struct('logv',false,'mu',true,'A',true,'eta',true);
Ignore = struct('A',false,'eta',false);
Options = struct('v0',v0,'mu0',mu0,'A0',A0,'eta0',eta0,'ts',T, ...
    'Fix',Fix,'Ignore',Ignore);

P(iFit) = tdnoisefit(X,Options);
v0 = P(iFit).var;
iFit = iFit + 1;

%% Fit for all parameters

Fix = struct('logv',false,'mu',false,'A',false,'eta',false);
Ignore = struct('A',false,'eta',false);
Options = struct('v0',v0,'mu0',mu0, ...
    'A0',A0,'eta0',eta0,'ts',T,'Fix',Fix,'Ignore',Ignore);

[P(iFit),nllmin,Diagnostic] = tdnoisefit(X,Options);

%% Compare model to measurements

vEst = P(end).var;
muEst = P(end).mu;
AEst = P(end).A;
etaEst = P(end).eta;

vErr = Diagnostic.Err.var;
muErr = Diagnostic.Err.mu;
AErr = Diagnostic.Err.A;
etaErr = Diagnostic.Err.eta;

zeta = zeros(N,M);
S = zeros(N,N,M);
for m = 1:M
    S(:,:,m) = shiftmtx(P(end).eta(m),N,T);
    zeta(:,m) = P(end).A(m)*S(:,:,m)*P(end).mu;
end

Dmu = D*P(end).mu;
valpha = P(end).var(1);
vbeta = P(end).var(2)*P(end).mu.^2;
vtau = P(end).var(3)*(Dmu).^2;
vtot = valpha + vbeta + vtau;

delta = (X - zeta)./sqrt(vtot);

valphalow = P(end).var(1) - vErr(1);
vbetalow = (P(end).var(2) - vErr(2))*P(end).mu.^2;
vtaulow = (P(end).var(3) - vErr(3))*(Dmu).^2;
vtotlow = valphalow + vbetalow + vtaulow;

valphahigh = P(end).var(1) + vErr(1);
vbetahigh = (P(end).var(2) + vErr(2))*P(end).mu.^2;
vtauhigh = (P(end).var(3) + vErr(3))*(Dmu).^2;
vtothigh = valphahigh + vbetahigh + vtauhigh;

Xadjusted = airscancorrect(X,P(end));

[hks,pks,ksstat,cvks] = kstest(delta(:));
[had,pad,adstat,cvad] = adtest(delta(:));
[hrun,prun,Runstat] = runstest(delta(:));

%% Summarize results

figure('Name','Data')
plot(t,X);
xlabel('t (ps)')
ylabel('X (pA)')

figure('Name','Residual plot')
plot(t,delta);
xlabel('t (ps)')
ylabel('(X - \zeta)/\sigma')

figure('Name','Normal distribution plot');
normplot(delta(:));
xlabel('(X - \zeta)/\sigma')

figure('Name','Noise model comparison');
Ln(1) = plot(t,std(Xadjusted,0,2),'k-', ...
    'DisplayName','Measurement');
hold on
Ln(2) = plot(t,sqrt(vtot),'r-', ...
    'DisplayName','Noise model');
Ln(3) = plot(t,sqrt(vbeta),'g-', ...
    'DisplayName','\sigma_{\beta}\mu');
Ln(4) = plot(t,sqrt(vtau),'b-', ...
    'DisplayName','\sigma_{\tau}(d\mu/dt)');
% Ln(3) = plot(t,sqrt(vtotlow),'r-.', ...
%     'DisplayName','Bound');
% Ln(4) = plot(t,sqrt(vtothigh),'r-.', ...
%     'DisplayName','Bound');

xlabel('t (ps)')
ylabel('\sigma(t)')
legend(Ln(1:4))
%%
figure('Name','std + Data')
plot(t,[X,10*std(Xadjusted,0,2)])
xlabel('t (ps)')
ylabel('Amp (pA)')
%%
fprintf('sigma_alpha^2: %.4g\n', vEst(1));
fprintf('Bias corrected: %.4g\n',vEst(1)*(M/(M-1))*(N/(N-1)));
fprintf('Standard error: %.4g\n\n',vErr(1));
fprintf('\n')

fprintf('sigma_beta^2: %.4g\n', vEst(2));
fprintf('Bias corrected: %.4g\n',vEst(2)*(M/(M-1))*(N/(N-1)));
fprintf('Standard error: %.4g\n\n',vErr(2));
fprintf('\n')

fprintf('sigma_tau^2: %.4g\n', vEst(3));
fprintf('Bias corrected: %.4g\n',vEst(3)*(M/(M-1))*(N/(N-1)));
fprintf('Standard error: %.4g\n\n',vErr(3));
fprintf('\n')

fprintf('Kolmogorov-Smirnov test (alpha = 5%%): %d, p = %g\n',hks, pks);
fprintf('Anderson-Darling test (alpha = 5%%): %d, p = %g\n',had, pad);
fprintf('Run test (alpha = 5%%): %d p = %g\n', hrun, prun);

%% Stop timer
toc(tStart)