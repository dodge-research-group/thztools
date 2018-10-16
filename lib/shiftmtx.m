function h = shiftmtx(tau,n,ts)
%SHIFTMTX computes the transfer matrix for a continuous-time shift
%
% Syntax:   H = shiftmtx(tau,n,ts)
%
% Description:
%
% SHIFTMTX computes the n by n transfer matrix for a continuous-time shift
% tau
%  
% Inputs:
%   tau   Input parameters for the function
%   n       Number of time samples
%   ts      Sampling time
%
% Outputs: 
%   H       transfer matrix with size (n,n)   
%


validateattributes(tau,{'double'},{'scalar'})
validateattributes(n,{'numeric'},{'scalar','positive'})
validateattributes(ts,{'numeric'},{'scalar','positive'})

% Fourier method
f =  fftfreq(n,ts);
w = 2*pi*f;

% Use MATLAB's +iwt convention
imp = real(ifft(exp(-1i*w*tau)));
    
h = toeplitz(imp,circshift(flipud(imp),1));

% % Trigonometric interpolation method
% if rem(n,2)
%     imp = diric(2*pi*((0:n-1)'*ts-tau)/(n*ts),n);
% else
%     imp = diric(2*pi*((0:n-1)'*ts-tau)/(n*ts),n)...
%         .*cos(pi*((0:n-1)'*ts-tau)/(n*ts));
% end
% 
% h = toeplitz(imp,circshift(flipud(imp),1));

end