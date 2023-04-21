import math
import numpy as np
import scipy
from scipy import linalg
from thztoolsPY.shiftmtx import shiftmtx
from package_example import SI

def airscancorrect(x, param):
    [n, m] = x.shape

    # Parse parameter structure
    pfields = param.keys()
    if 'a' in pfields and param.get('a') is not None:
        a = param.get('a').T
    else:
        a = np.ones((m, 1))
        # Ignore.A = true
    if 'eta' in pfields and param.get('eta') is not None:
        eta = param.get('eta')
    else:
        eta = np.zeros((m, 1))
    if 'ts' in pfields:
        ts = param['ts']
    else:
        ts = 1

    xadj = np.zeros((n, m))
    for i in np.arange(m):
        s = shiftmtx(-eta[i], n, ts)
        xadj[:, i] = s @ (x[:, i] / a[i])

    return xadj


#######################################################

def sigmamu(sigma, mu, t):
    return np.sqrt(noisevar(sigma, mu, t))

########################################################

def fftfreq(n, t):
    if n % 2 == 1:
        f = np.fft.fftfreq(n, t)
    else:
        f = np.fft.fftfreq(n, t)
        f[int(n / 2)] = -f[int(n / 2)]

    return f


##########################################################

def noisevar(sigma, mu, t):
    n = len(mu)
    w = 2 * np.pi * fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j * w * np.fft.fft(mu)))

    return sigma[0] ** 2 + (sigma[1] * mu) ** 2 + (sigma[2] * mudot) ** 2


###########################################################

def tdtf(fun, theta, n, ts):
    # compute the transfer function over positive frequencies
    fs = 1 / (ts * n)
    fp = fs * np.arange(0, math.floor((n - 1) / 2 + 1))
    wp = 2 * np.pi * fp
    tfunp = fun(theta, wp)

    # The transfer function is Hermitian, so we evaluate negative frequencies
    # by taking the complex conjugate of the corresponding positive frequency.
    # Include the value of the transfer function at the Nyquist frequency for
    # even n.
    if n % 2 != 0:
        tfun = np.concatenate((tfunp, np.conj(np.flipud(tfunp[1:]))))

    else:
        wny = np.pi * n * fs
        print('tfunp', tfunp)
        tfun = np.concatenate((tfunp, np.conj(np.concatenate((fun(theta, wny), np.flipud(tfunp[1:]))))))

    # Evaluate the impulse response by taking the inverse Fourier transform,
    # taking the complex conjugate first to convert to ... +iwt convention

    imp = np.real(np.fft.ifft(np.conj(tfun)))
    h = linalg.toeplitz(imp, np.roll(np.flipud(imp), 1))

    return h


############################################################

def costfun(fun, mu, theta, xx, yy, sigma_alpha, sigma_beta, sigma_tau, ts):
    sigma = [sigma_alpha, sigma_beta, sigma_tau]

    n = len(xx)
    h = tdtf(fun, theta, n, ts)

    psi = h * mu

    vmu = np.diag(noisevar(sigma, mu, ts))
    vpsi = np.diag(noisevar(sigma, psi, ts))

    ivmu = np.diag(1 / noisevar(sigma, mu, ts))
    ivpsi = np.diag(1 / noisevar(sigma, psi, ts))

    # compute the inverse covariance matrices for xx and yy
    ivx = np.diag(1 / noisevar(sigma, xx, ts))
    ivy = np.diag(1 / noisevar(sigma, xx, ts))

    # compute the cost function
    # Note: sigma-mu and sigma-psi both have determinants below the numerical
    # precision, so we multiply them by the constant matrices isigmaxx and
    # isigmayy to improve numerical stability
    k = np.log(np.linalg.det(ivx * vmu)) + np.log(np.linalge.det(ivy * vpsi)) + \
        (xx - mu).T * ivmu * (xx - mu) + (yy - psi).T * ivpsi * (yy - psi)

    return k


####################################################

def costfunlsq(fun, theta, xx, yy, sigmax, sigmay, wfft):
    wfft1 = np.array([wfft])
    n = len(sigmax)
    # n =np.shape(wfft)[0]
    h = np.conj(fun(theta, wfft))
    if n % 2 == 0:
        kny = n // 2
        h[kny] = np.real(h[kny])

    ry = yy - np.real(np.fft.ifft(np.fft.fft(xx) * h))
    vy = np.diag(sigmay ** 2)

    htilde = np.fft.ifft(h)

    uy = np.zeros((n, n))
    for k in np.arange(n):
        a = np.reshape(np.roll(htilde, k), (n, 1))
        b = np.reshape(np.conj(np.roll(htilde, k)), (1, n))
        uy = uy + np.real(np.dot(a, b)) * sigmax[k] ** 2
        # uy = uy + np.real(np.roll(htilde, k-1) @ np.roll(htilde, k-1).T) @ sigmax[k]**2

    w = np.dot(np.eye(n), scipy.linalg.inv(scipy.linalg.sqrtm(uy + vy)))
    res = np.dot(w, ry)

    return res


#########################################################
def costfunwofflsq(fun, theta, xx, yy, alpha, beta, covx, covy, ts):
    n = len(xx)
    h = tdtf(fun, theta, n, ts)

    icovx = np.eye(n) / covx
    icovy = np.eye(n) / covy

    m1 = np.eye(n) + (covx * h.h * icovy * h)
    im1 = np.eye(n) / m1
    m2 = (xx - alpha) + covx * h.h * icovy * (yy - beta)
    im1m2 = im1 * m2
    hm1invm2 = h * im1m2

    res = [scipy.linalg.sqrtm(icovx) * (xx - alpha - im1m2), scipy.linalg.sqrtm(icovy) * (yy - beta - hm1invm2)]

    return res


##################################################

def epswater(f, t=25):
    # Frequency conversion to Hz
    f = f * 1e12

    # Define relaxation parameters
    a = np.array([79.23882, 3.815866, 1.634967])
    b = np.array([0.004300598, 0.01117295, 0.006841548])
    c = np.array([1.382264e-13, 3.510354e-16, 6.30035e-15])
    d = np.array([652.7648, 1249.533, 405.5169])
    tc = 133.1383

    # Define resonance parameters
    p0 = 0.8379692
    p = np.array([-0.006118594, -0.000012936798, 4235901000000, -14260880000,
                  273815700, -1246943, 9.618642e-14, 1.795786e-16, -9.310017e-18,
                  1.655473e-19, 0.6165532, 0.007238532, -0.00009523366, 15983170000000,
                  -74413570000, 497448000, 2.882476e-14, -3.142118e-16, 3.528051e-18])

    # Compute temperature - dependent functions
    eps0 = 87.9144 - 0.404399 * t + 9.58726e-4 * t ** 2 - 1.32802e-6 * t ** 3
    delta = a * np.exp(-b * t)
    tau = c * np.exp(d / (t + tc))

    delta4 = p0 + p[0] * t + p[1] * t ** 2
    f0 = p[2] + p[3] * t + p[4] * t ** 2 + p[5] * t ** 3
    tau4 = p[6] + p[7] * t + p[8] * t ** 2 + p[9] * t ** 3
    delta5 = p[10] + p[11] * t + p[12] * t ** 2
    f1 = p[13] + p[14] * t + p[15] * t ** 2
    tau5 = p[16] + p[17] * t + p[18] * t ** 2

    # Put it all together
    epsilonr = (eps0 + 2 * 1j * np.pi * f * (delta[0] * tau[0] / (1 - 2 * 1j * np.pi * f * tau[0])
                                             + delta[1] * tau[1] / (1 - 2 * 1j * np.pi * f * tau[1])
                                             + delta[2] * tau[2] / (1 - 2 * 1j * np.pi * f * tau[2]))
                + 1j * np.pi * f * (delta4 * tau4 / (1 - 2 * 1j * np.pi * tau4 * (f0 + f))
                                    + delta4 * tau4 / (1 + 2 * 1j * np.pi * tau4 * (f0 - f)))
                + 1j * np.pi * f * (delta5 * tau5 / (1 - 2 * 1j * np.pi * tau5 * (f1 + f))
                                    + delta5 * tau5 / (1 + 2 * 1j * np.pi * tau5 * (f1 - f))))

    return epsilonr


##########################################

def jpcrateeq(t, theta):
    n = theta[0]
    v = theta[1]
    p = theta[2]

    taus = 0.1
    tauc = 0.4
    taur = 100
    taul = 0.05 / np.sqrt(2 * np.log(2))
    mstar = 0.063
    epsilonr = 12.9
    eta = 3
    g0 = 2e8
    ebias = 100 / 400

    e_over_m = 1e-12 * SI.qe / SI.me
    e_over_eps0 = 1e6 * SI.qe / SI.eps0

    dvdt = -v / taus - e_over_m * ebias / mstar + e_over_m * e_over_eps0 * p / (mstar * eta * epsilonr * taur)
    dpdt = -p / taur - n * v
    dndt = -n / tauc + g0 * np.exp(-(t / taul) ** 2 / 2)

    return [dndt, dvdt, dpdt]

################################################

def noisevar(sigma, mu, t):
    n = len(mu)
    w = 2 * np.pi * fftfreq(n, t)
    mudot = np.real(np.fft.ifft(1j * w * np.fft.fft(mu)))

    return sigma[0] ** 2 + (sigma[1] * mu) ** 2 + (sigma[2] * mudot) ** 2