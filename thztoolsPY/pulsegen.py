import numpy as np

def pulsegen(N, t0, w, A, T):
    t = T*np.arange(0, N)
    tt = (t-t0)/w

    y = A*(1-2*tt**2)*np.exp(-tt**2)

    return [y, t]




