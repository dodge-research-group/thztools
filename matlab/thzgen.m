function [y,t] = thzgen(N,T,t0,varargin)
%THZGEN generate a terahertz pulse
%
% Syntax:   y = thzgen(N,T,t0)
%           [y,t] = thzgen(N,T,t0)
% 
% Description:
% 
% thzgen(N,T,t0) generates terahertz pulse with N points at sampling
% interval T and centered at t0.
% For use in tests of time-domain analysis.
% 
% Inputs:
%   N       number of sampled points
%   T       sampling time
% 	t0      pulse center
% 
% Outputs:
%   y       signal      [a.u.]
%   t       timebase    [T]
% 
% 

default_A = 1;
default_taur = 0.3;
default_tauc = 0.1;
default_taul = 0.05/sqrt(2*log(2));

validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);

Input = inputParser;

addRequired(Input,'N',validScalarPosNum)
addRequired(Input,'T',validScalarPosNum)
addRequired(Input,'t0',validScalarPosNum)

addParameter(Input,'A',default_A,validScalarPosNum)
addParameter(Input,'taur',default_taur,validScalarPosNum)
addParameter(Input,'tauc',default_tauc,validScalarPosNum)
addParameter(Input,'taul',default_taul,validScalarPosNum)

parse(Input,N,T,t0,varargin{:})

A = Input.Results.A;
taur = Input.Results.taur;
tauc = Input.Results.tauc;
taul = Input.Results.taul;


f = fftfreq(N,T);
w = 2*pi*f;

L = exp(-(w*taul).^2/2)/sqrt(2*pi*taul^2);
R = 1./(1/taur - 1i*w) - 1./(1/taur + 1/tauc - 1i*w);
S = -1i*w.*(L.*R).^2.*exp(1i*w*t0);

t=T*(0:N-1);
t=t(:);

y = real(ifft(conj(S)));
y = A*y/max(y);
end