import scipy 
from scipy.linalg import sqrtm

def costfunlsq(fun,theta,xx,yy,sigmax,sigmay,wfft):
    
    N = len(sigmax)
    H = np.conj(fun(theta,wfft))
    
    if N % 2 == 0:
        kNy = N/2
        H[kNy+1] = np.real(H[kNy+1])
        
    ry = yy - np.real(np.fft.ifft(np.fft(xx)*H))
    Vy = np.diag(sigmay**2)
    
    Htilde = np.fft.ifft(H)
    
    Uy = np.zeros(N)
    
    W = np.eye(N)/scipy.linalg.sqrtm(Uv + Vy)
    
    res = W*ry