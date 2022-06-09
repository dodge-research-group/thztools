import numpy as np

import SI

def jpcrateeq(t, theta):
    # JPCRATEEQ Rate equations for photoconducting antenna current

    n = theta[0]
    v = theta[1]
    p = theta[2]

    taus = 0.1
    tauc = 0.4
    taur = 100
    taul = 0.05/np.sqrt(2*np.log(2))
    mstar = 0.063
    epsilonr = 12.9
    eta = 3
    g0 = 2e8
    Ebias = 100/400

    e_over_m = 1e-12*SI.qe/SI.me
    e_over_eps0 = 1e6*SI.qe/SI.eps0

    dvdt = -v/taus - e_over_m*Ebias/mstar \
        + e_over_m*e_over_eps0*p/(mstar*eta*epsilonr*taur)
    dpdt = -p/taur - n*v
    dndt = -n/tauc + g0*np.exp(-(t/taul)**2/2)

    return [dndt, dvdt, dpdt]
