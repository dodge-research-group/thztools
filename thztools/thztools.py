import numpy as np
from numpy.fft import fft, ifft
from scipy import constants
from scipy.linalg import toeplitz

def fftfreq(n, ts):
    """
    Generates array frequencies for the discrete Fourier transform
    Parameters
    ----------
    n: int
      Number of points in the array.
    ts: float
      Sampling time.

    Returns
    -------
    array_like float

    Notes
    -----
    The convention used here is different from the NumPy fft module when n is even. Here,
    the Nyquist frequency is positive, and in the NumPy fft module, the Nyquist frequency
    is negative.
    """
    x = np.arange(n)
    xsub = x - np.floor((n - 1) / 2)
    kcirc = np.roll(xsub, np.ceil((n + 1) / 2).astype(int))
    f = kcirc / (n * ts)

    return f


def noisevar(sigma, mu, ts):
    """
    Compute the time-domain noise variance.
    Parameters
    ----------
    sigma: array_like float
        Array with 3 elements.
        sigma[0]: Additive noise amplitude (units of mu)
        sigma[1]: Multiplicative noise amplitude (dimensionless)
        sigma[2]: Timebase noise amplitude (units of ts)
    mu: array_like float
        Signal array.
    ts: float
        Sampling time.

    Returns
    -------
    array_like float
        Noise variance for each element of mu.
    """
    # Use FFT to compute derivative of mu for timebase noise
    n = int(len(mu))
    w = 2 * np.pi * fftfreq(n, ts)
    muf = fft(mu)
    mudotifft = ifft(1j * w * muf)
    mudot = np.real(mudotifft)

    # Compute noise variance
    cov_mu = sigma[0] ** 2 + (sigma[1] * mu)**2 + (sigma[2] * mudot)**2
    return cov_mu


def noiseamp(sigma, mu, T):
    sigmamu = np.sqrt(noisevar(sigma, mu, T))  # Are we square rooting the matrix, or the elements individually
    return sigmamu


def jpcrateeq(t, theta):




    n = theta[0]
    v = theta[1]
    p = theta[2]



    e_over_m = .000000000001*constants.e/constants.m_e
    e_over_eps0 = 100000*constants.e/constants.epsilon_0

    dvdt = -v/taus - e_over_m*Ebias/mstar + e_over_m*e_over_eps0*p/(mstar*eta*epsilonr*taur)
    dpdt = -p/taur - n*v
    dndt = -n/tauc + g0*np.exp(-(t/taul)**2/2)

    dthetadt = [dndt, dvdt, dpdt]
    print(dthetadt)
    return dthetadt


def thzgen(N, T, t0):
    f = thztools.fftfreq(N, ts)
    w = 2 * np.pi * f

    # Computing L
    Lsq = np.square(w * taul * -1)  # (-(w*taul).^2/2)

    L = np.exp(Lsq / 2) / np.sqrt(2 * np.pi * (taul ** 2))

    # computing R

    iw = -1j * w
    invw = 1 / iw

    R = 1 / (1 / taur - iw) - 1 / (1 / taur + 1 / tauc - iw)

    # computing S

    S = iw * (L * R) ** 2 * np.exp(iw * t0)

    #  computing timebase matrix by multiplying sample number and the sampling time
    #  first create a matrix with N elements, then add one so that indexing begins at 1

    tm = np.arange(13)
    tm = tm + 1
    t = T * tm

    # y calcuations

    Sconj = np.conj(S)
    Sifft = np.fft.ifft(Sconj)
    y = np.real(Sifft)
    y = A * y / max(y)

    return y


def shiftmtx(tau, n, ts):
    f = fftfreq(n, ts)
    w = 2 * np.pi * f

    imp = np.real(ifft(np.exp(-1j * w * tau)))

    h = toeplitz(imp, np.roll(imp[::-1], 1))

    return h


def tdtf(fun, theta, N, ts):
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
    fs = 1 / (ts * N)
    p = np.arange((N - 1) / 2)
    fp = fs * p
    wp = 2 * np.pi * fp
    tfunp = fun(theta, wp)

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
    flipc = np.conj(flipt)

    #  The transfer function is Hermitian, so the frequencies after the transformation
    #  are the same magnitude but negative

    if N % 2 != 0:
        tfun = np.concatenate(tfunp, flipc)
    else:
        wny = np.pi * N * fs
        tfun = np.concatenate(tfunp, np.conj(fun(theta, wny)), flipc)

    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = toeplitz(imp, np.roll(np.flipud(imp), 1))
    
    
def pulsegen( N, t0, w, A, T ):


    t = T*(np.arange(N))
    tt = (t - t0)/w

    y = A*(1-(2*tt)**2)*np.exp(-tt**2)
    print(y)
    return y, t
