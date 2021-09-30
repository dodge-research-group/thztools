function res = costfunlsq(fun,theta,xx,yy,sigmax,sigmay,wfft)

N = length(sigmax);
H = conj(fun(theta, wfft));
if rem(N,2)==0
    kNy = N/2;
    H(kNy+1) = real(H(kNy+1));
end

ry = yy - real(ifft(fft(xx).*H));
Vy = diag(sigmay.^2);

Htilde = ifft(H);

Uy = zeros(N);
for k = 1:N
    Uy = Uy + real(circshift(Htilde,k-1)...
        *(circshift(Htilde,k-1)'))*sigmax(k)^2;
end

W = eye(N)/sqrtm(Uy + Vy);

res = W*ry;
end