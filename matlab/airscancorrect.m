function Xadj = airscancorrect(X,Param)
%AIRSCANCORRECT rescales and shifts data matrix
%
% Syntax:   Xadj = airscancorrect(X,Param)
%
% Description:
%
% AIRSCANCORRECT rescales and shifts each column of the data matrix X,
% assuming that each column is related to a common signal by an amplitude A
% and a delay eta.
%  
% Inputs:
%   X       Data matrix
%   Param   Parameter structure, including:
%       .A      Amplitude vector        [Mx1 double]
%       .eta    Delay vector            [Mx1 double]
%       .ts     Sampling time           [double]
%
% Outputs: 
%   Xadj	Adjusted data matrix
%

% Parse function inputs
[N,M] = size(X);
validateattributes(X,{'double'},{'2d'})
validateattributes(Param,{'struct'},{'nonempty'})

% Parse parameter structure
Pfields = fieldnames(Param);
if ismember('A',Pfields) && ~isempty(Param.A)
    A = Param.A(:);
    validateattributes(A,{'double'},{'vector','numel',M})
    Ignore.A = false;
else
    A = ones(M,1);
    Ignore.A = true;
end
if ismember('eta',Pfields) && ~isempty(Param.eta)
    eta = Param.eta(:);
    validateattributes(eta,{'double'},{'vector','numel',M})
    Ignore.eta = false;
else
    eta = zeros(M,1);
    Ignore.eta = true;
end
if ismember('ts',Pfields)
    ts = Param.ts;
    validateattributes(ts,{'double'},{'scalar'})
else
    ts = 1;
    warning('TDNLL received Param structure without ts field; set to one')
end

Xadj = zeros(N,M);
for m = 1:M
    S = shiftmtx(-eta(m),N,ts);
    Xadj(:,m) = S*X(:,m)/A(m);
end

end