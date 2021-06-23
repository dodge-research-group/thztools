'''
1) Variable definitions
2) Create function called noisevar
    Inputs: Sigma (delays)
            mu 
            T
    Output: Vmu
3) Get length of MU
4) Use fftfreq function to create frequency vector
5) Convert frequencies to angular frequencies
6) Convert angular frequencies to complex numbers
7) complete fast fourier transform on mu get amplitude by frequency
8) Multiply complex angular frequency vector * frequency base of mu
9) Take inverse fft of the resulting product
10) Take only real values of resulting vector over time base
11) Calculate Vmu
    '''
import numpy as np
from testfreq import fftfreq

mu = [ 1, 2 ,3 ,4, 5, 6, 7, 8, 9, 10, 11]
sigma = [.3, .4 ,.5]
T = .05
ts = .05
N = 38
 
fs = 1/(ts*N)




def noisevar(sigma, mu, T):

    N = int(len(mu))
   
    
    w = 2*np.pi*np.fft.fftfreq(N,T)
    
    
    jw = 1j*w
    
    
    muf = np.fft.fft(mu)
     
    jwmuf = jw*muf
    
    
    
    mudotifft  = np.fft.ifft(jwmuf)
    mudot = np.real(mudotifft)
    
     
    Vmu = sigma[0]**2 + (np.square(np.multiply(sigma[1],mu))) +  (np.square(np.multiply(sigma[2],mudot)))
    print(Vmu)
    return Vmu

fftfreq(N,ts)
noisevar(sigma,mu,T)

 