function [y,t] = pulsegen(N,t0,w,A,T)
%PULSEGEN generates a short pulse of temporal width w centered at t0.
%
% Syntax:   y = pulsegen(N,t0,w,A,T)
%           [y,t] = pulsegen(N,t0,w,A,T)
% 
% Description:
% 
% pulsegen(N,t0,w,A,T) generates a short pulse of width w centered at t0.
% For use in tests of time-domain analysis.
% 
% Inputs:
%   N       number of sampled points
% 	t0      pulse center
%   w       pulse width
%   A       pulse amplutide
%   T       sampling time
% 
% Outputs:
%   y       signal      [a.u.]
%   t       timebase    [T]
% 
% 

t=T*(0:N-1);
t=t(:);
tt = (t-t0)/w;

y = A*(1-2*tt.^2).*exp(-tt.^2);