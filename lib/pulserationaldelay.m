function signalOut = pulserationaldelay(signalIn,Parms)
%PULSERATIONALDELAY applies rational transfer function with delay
%
% Syntax:	signalOut = pulsepolyratiodelay(signalIn,Parms)
% 
% Description:
% 
% pulserationaldelay calculates time-domain output with a rational transfer
% function together with a delay:
%
%           b(1) + b(2)*s + b(3)*s^2 + ... + b(nb)*s^(nb-1)
%   t =    ------------------------------------------------- exp(-s*eta),
%           a(1) + a(2)*s + a(3)*s^2 + ... + a(na)*s^(na-1)
% 
%   where s = i*w and w is the scaled angular frequency,
%   ranging over (-pi,pi). If Parms is a structure array, the transfer
%   function is a sum of rational functions, in which each term in the sum
%   is formed from the parameters contained in an array element.
%
% Inputs:
%   signalIn	input data vector, arbitrary units
%   Parms       Structure array of parameters:
%       .a          Denominator coefficients
%       .b          Numerator coefficients
%       .eta        Delay
%    
% 
% Outputs:
%   signalOut	output data vector, arbitrary units
% 
% 
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%

% Author: J. Steven Dodge, 2005
% Physics Dept., Simon Fraser University
% Website: http://www.physics.sfu.ca/people/profiles/jsdodge
%
% $Id$

%------------- BEGIN CODE --------------

[N,c] = size(signalIn);
Nmax = floor((N-1)/2);
nTerm = length(Parms);

if c~=1
    warning('Multiple columns in data arrays, operating on first')
    signalIn=signalIn(:,1);
end

f = (1:Nmax)/N;
w = 2*pi*f;
w = w(:);
s = 1i*w;

% Fourier transform signal
signalInFt = fft(signalIn);
in = signalInFt(2:Nmax + 1);

% Initialize transfer function array
tFun = zeros(Nmax,1);

% Sum over terms
for iTerm = 1:nTerm

    na=length(Parms(iTerm).a)-1;
    nb=length(Parms(iTerm).b)-1;

    % Form the numerator and denominator
    nab = max(na,nb);
    kVal = (0:nab);
    vanderS = (s(:,ones(1,nab+1)).^kVal(ones(Nmax,1),:));

    denominator = vanderS(:,1:na+1)*Parms(iTerm).a;
    numerator   = vanderS(:,1:nb+1)*Parms(iTerm).b;

    tFun = tFun + (numerator./denominator).*exp(-Parms(iTerm).eta*s);
end
% Run through rational transfer function
out = tFun.*in;

if (rem(N,2)==0)
    signalOut = ...
        real(ifft([0;out;0;conj(flipud(out))]));
else
    signalOut = ...
        real(ifft([0;out;conj(flipud(out))]));
end
%------------- END OF CODE --------------