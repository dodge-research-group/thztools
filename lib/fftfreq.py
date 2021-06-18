 


import numpy as np
import math 

N = 11     #number of samples
ts = .05   #sampling time
fs = 1/(ts*N)   #sampling frequency

def fftfreq(N,ts):
  x = np.arange(N+1)  #creates a matrix N+1 length
  xsub = x - np.ceil(N/2)  #subtracts N/2 rounded up from all values in x
  kcirc = np.roll(xsub,(np.floor(N/2)+1))   #this shifts the zero frequency to first positioon
  f = kcirc*fs  #creates the frequency vector
  fp = np.transpose(f)  #turns it into a N x 1 matrix
  
  return fp
