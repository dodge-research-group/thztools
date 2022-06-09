import numpy as np

def pulsegen(N, t0, w, A, T):
    """
    pulsegen generates a short pulse of temporal width w centered at t0

        Parameters:
            N : number of sampled points
            t0 : pulse center
            w : pulse width
            A : pulse amplitude
            T : sampling time

        Returns:
            list of signal data, and timebase data [y,t]
    """

    # Generates array of time data for specified sample points N, and
    # sampling time T
    t = T*np.arange(0, N)
    tt = (t-t0)/w

    # computes amplitude of pusle over time data
    y = A*(1-2*tt**2)*np.exp(-tt**2)

    return [y, t]




