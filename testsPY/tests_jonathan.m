% % %%TDNOISEFIT %%%%%%
% 
% Data = dataToptica;
% nDir = length(Data);
% 
% %Run loop backwards to preallocate structure array upon first iteration
% Data = Data(nDir:-1:1);
% for iDir = length(Data):-1:1
%     t = Data(iDir).t;
%     X = Data(iDir).X;
%     [~, dirName] = fileparts(Data(iDir).Dir);
%     fprintf("%s\n", repmat('-',1,50));
%     fprintf("%s\n", repmat('-',1,50));
%     fprintf('Fit noise model to data in %s\n', dirName)
%     [Output(iDir), Options, P] = noisefitshow(t, X, replace(dirName, '_', ' '));
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5


%tdnoisefit_data = struct('X', X,  't', t, 'Options', Options, 'Output', Output);

%save tdnoisefit_test_data.mat -v7.3 tdnoisefit_data
% vEst = P(end).var;
% muEst = P(end).mu;
% AEst = P(end).A;
% etaEst = P(end).eta;
% 
% vEstStar = vEst*50/(50-1);
% 
% sigmaAlphaStar = sqrt(vEstStar(1));
% sigmaBetaStar = sqrt(vEstStar(2));
% sigmaTauStar = sqrt(vEstStar(3));
% 
% fun = @(theta, w) -1i*w;
% D = tdtf(fun, 0, 1001, 0.05);
% Dmu = D*P(end).mu;
% 
% vAlphaStar = vEstStar(1);
% vBetaStar = vEstStar(2)*P(end).mu.^2;
% vTauStar = vEstStar(3)*(Dmu).^2;
% vTotStar = vAlphaStar + vBetaStar + vTauStar;
% 
% zeta = zeros(1001, 50);
% S = zeros(1001, 1001, 50);
% for m = 1:50
%     S(:,:,m) = shiftmtx(etaEst(m), 1001, 0.05);
%     zeta(:,m) = AEst(m)*S(:,:,m)*muEst;
% end
% 
% delta = (X - zeta)./sqrt(vTotStar);
% 
% size(vTotStar);
% size(zeta);
% 
% size(delta)


P