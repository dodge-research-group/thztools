function res = costfunlsq(fun,theta,xx,yy,covx,covy,ts)

n = length(xx);
H = tdtf(fun,theta,n,ts);

icovx = eye(n)/covx;
icovy = eye(n)/covy;

M1 = eye(n) + (covx*H'*icovy*H);
iM1 = eye(n)/M1;
M2 = xx + covx*H'*icovy*yy;
iM1M2 = iM1*M2;
HiM1M2 = H*iM1M2;

res = [sqrtm(icovx)*(xx-iM1M2); sqrtm(icovy)*(yy-HiM1M2)];
end