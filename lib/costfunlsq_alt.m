function res = costfunlsq_alt(fun,theta,xx,yy,sigmax,sigmay,wfft)

N = length(sigmax);
H = conj(fun(theta, wfft));
kNy = floor(N/2);
H(kNy+1) = real(H(kNy+1));

rx = xx - real(ifft(fft(yy)./H));
Vx = diag(sigmax.^2);

Htildeinv = ifft(1./H);

Ux = zeros(N);
for k = 1:N
    Ux = Ux + real(circshift(Htildeinv,k-1)...
        *(circshift(Htildeinv,k-1)'))*sigmay(k)^2;
end

W = eye(N)/sqrtm(Vx + Ux);

res = W*rx;
end