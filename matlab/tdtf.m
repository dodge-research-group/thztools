function h = tdtf(fun,theta,N,ts)
%TDTF computes the transfer matrix for a given function
%
% Syntax:   h = tdtf(fun,theta,N,ts)
%
% Description:
%
% TDTF computes the N-by-N transfer matrix for the given function fun with 
% input parameter theta. Note that the transfer function should be written
% using the physicist's -iwt convention instead of MATLAB's +iwt
% convention, and that it should be in the format fun(theta,w), where theta
% is a vector of the function parameters. The transfer function must be
% Hermitian.
%  
% Inputs:
%   fun     Transfer function, in the form fun(theta,w), -iwt convention
%   theta   Input parameters for the function
%   N       Number of time samples
%   ts      Sampling time
%
% Outputs: 
%   h       transfer matrix with size (N,N)   
%


validateattributes(fun,{'function_handle'},{'scalar'})
validateattributes(theta,{'double'},{'vector'})
validateattributes(N,{'numeric'},{'scalar','positive'})
validateattributes(ts,{'numeric'},{'scalar','positive'})

% Compute the transfer function over positive frequencies
fs = 1/(ts*N);
fp = fs*(0:floor((N-1)/2));
fp = fp(:);
wp = 2*pi*fp;

tfunp = fun(theta,wp);

% The transfer function is Hermitian, so we evaluate negative frequencies
% by taking the complex conjugate of the correponding positive frequency.
% Include the value of the transfer function at the Nyquist frequency for
% even N.
if rem(N,2)~=0
    tfun = [tfunp; conj(flipud(tfunp(2:end)))];
else
    wNy = pi*N*fs;
    tfun = [tfunp; conj([fun(theta,wNy);flipud(tfunp(2:end))])];
end

% Evaluate the impulse response by taking the inverse Fourier transform,
% taking the complex conjugate first to convert to MATLAB's +iwt convention
imp = real(ifft(conj(tfun)));
h = toeplitz(imp,circshift(flipud(imp),1));
end