% %%TDNOISEFIT %%%%%%

Data = dataToptica;
nDir = length(Data);

%Run loop backwards to preallocate structure array upon first iteration
Data = Data(nDir:-1:1);
for iDir = length(Data):-1:1
    t = Data(iDir).t;
    X = Data(iDir).X;
    [~, dirName] = fileparts(Data(iDir).Dir);
    fprintf("%s\n", repmat('-',1,50));
    fprintf("%s\n", repmat('-',1,50));
    fprintf('Fit noise model to data in %s\n', dirName)
    [Output(iDir), Options, P] = noisefitshow(t, X, replace(dirName, '_', ' '));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5


%tdnoisefit_data = struct('X', X,  't', t, 'Options', Options, 'Output', Output);

%save tdnoisefit_test_data.mat -v7.3 tdnoisefit_data

P(4).var