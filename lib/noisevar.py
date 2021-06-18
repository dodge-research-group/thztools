import numpy as np

def noisevar(sigma, mu, T):

    N = len(mu)
    
    
    w = 2*np.pi*fftfreq(N,T)
    jw = 1j*w
    muf = np.fft(mu)
    jwmuf = jw*muf
   
    mudotifft  = np.fft.ifft(Sconj)
    mudot = np.real(mudotifft)
    
    
    Vmu = sigma[0]**2 + (sigma[1]*mu)**2 +  (sigma[2]*mudot)**2
    
    return Vmu