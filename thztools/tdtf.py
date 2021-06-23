from scipy.linalg import toeplitz 
import numpy as np
theta = .1
N = 12
ts = .05
def tdtf(fun,theta,N,ts):
    '''
    
    TDTF computes the transfer matrix for a given function
    
    Inputs:
    
        fun     Transfer function, in the form fun(theta,w), -iwt convention
        theta   Input parameters for the function
        N       Number of time samples
        ts      Sampling time
    
    Outputs:
    
        h transfer matrix with size (N,N)

    '''
    
    # computing the transfer function over positive frequencies
    fs = 1/(ts*N)
    p = np.arange((N-1)/2)
    fp = fs*p
    wp = 2*np.pi*fp
    tfunp = fun(theta,wp)
    print(wp)
    
    '''
    From the original matlab code
    
    if rem(N,2)~=0
        tfun = [tfunp; conj(flipud(tfunp(2:end)))];
    else
        wNy = pi*N*fs;
        tfun = [tfunp; conj([fun(theta,wNy);flipud(tfunp(2:end))])];
    end
    ''' 
    tpsh = tfunp[1:end]
    flipt = np.flipud(tpsh)
    flipc = np.conj(fliptpsh)
    
    
    #  The transfer function is Hermitian, so the frequencies after the transformation 
    #  are the same magnitude but negative
    

    if N % 2 != 0:
        tfun  = np.concatenate(tfunp,flipt)
    else:
        wny = np.pi*N*fs
        twny = fun(theta,wNy)
        conjtwnh = np.concatenate(twny,fliptsh)
        fliptwny = np.concatenate(twny, fliptpsh)
        
        tfun = np.concatenate(tfunp, np.conj(fun(theta,wNy)),fliptpsh)
    
    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = toeplitz(imp, np.roll(np.flipud(imp),1))

tdtf(fun, theta, N , ts)
