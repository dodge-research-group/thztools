%% TDNOISEFIT %%%%%%

load TopticaNoiseOutput.mat
Data = dataMenlo; 

t = Data.t;
x = Data.X;

%Data is apodized, so restrict window
idx = (t >= 115) & (t <= 158);
t = t(idx);
x = x(idx,:);
[~, dirName] = fileparts(Data.Dir);
fileparts(dirName);
fprintf("%s\n", repmat('-',1,50));
fprintf("%s\n", repmat('-',1,50));
fprintf('Fit noise model to data in %s\n', dirName); 
Output = noisefitshow(t, x, 'Menlo Systems');





