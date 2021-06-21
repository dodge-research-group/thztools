import numpy as np


def fftfreq(N, ts):
  x = np.arange(N)  #creates a matrix N+1 length
  xsub = x - np.floor(N/2)  #subtracts N/2 rounded up from all values in x
  kcirc = np.roll(xsub, np.ceil(N/2).astype(int))   #this shifts the zero frequency to first position
  print("kcirc = {}".format(kcirc))
  f = kcirc/(N*ts)  #creates the frequency vector
  
  return f
