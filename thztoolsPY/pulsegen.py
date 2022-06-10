import numpy as np

def pulsegen(n, t0, w, a, t):
    t2 = t*np.arange(n)
    tt = (t2-t0)/w

    y = a*(1-2*tt**2)*np.exp(-tt**2)

    return [y, t]
