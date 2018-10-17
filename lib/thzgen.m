function [y,t] = thzgen(N,T,t0)
%THZGEN generate a terahertz pulse
%
% Syntax:   y = pulsegen(N,T,t0)
%           [y,t] = pulsegen(N,T,t0)
% 
% Description:
% 
% pulsegen(N,T,A,t0) generates terahertz pulse with N points at sampling
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

A = 1;
taur = 0.2;
tauc = 0.1;
taul = 0.05/sqrt(2*log(2));

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