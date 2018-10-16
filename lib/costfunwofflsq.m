function res = costfunwofflsq(fun,theta,xx,yy,alpha,beta,covx,covy,ts)

n = length(xx);
H = tdtf(fun,theta,n,ts);

icovx = eye(n)/covx;
icovy = eye(n)/covy;

M1 = eye(n) + (covx*H'*icovy*H);
iM1 = eye(n)/M1;
M2 = (xx - alpha) + covx*H'*icovy*(yy-beta);
iM1M2 = iM1*M2;
HM1invM2 = H*iM1M2;

res = [sqrtm(icovx)*(xx-alpha-iM1M2); sqrtm(icovy)*(yy-beta-HM1invM2)];
end