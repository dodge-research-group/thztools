import numpy as np
# from thztoolsPY.thzgen import thzgen
# from thztoolsPY.fftfreq import fftfreq
# from thztoolsPY.noiseamp import sigmamu
from thztoolsPY.thzgen import thzgen
from thztoolsPY.fftfreq import fftfreq
from thztoolsPY.noiseamp import sigmamu


def datamc(**kwargs):
    # Set defaults
    default_p = {
        "N": 256,
        "A": 1,  # Amplitude [nA]
        "T": 0.05,  # Sampling time [ps]
        "t0": 2.5,  # Peak pulse time [ps]
        "w": 0.25,  # Pulse width [ps]
        "sigmaAlpha": 1e-4,  # Additive noise amplitude [relative to peak]
        "sigmaBeta": 0.01,  # Multiplicative noise amplitude [-]
        "sigmaTau": 1e-3,  # Time base noise amplitude [ps]
        "Nmc": 500,
        "seed": 0
    }

    # Update defaults with user-defined parameters
    p = default_p.copy()
    p.update(kwargs)

    # Set constants
    n = p["N"]
    T = p["T"]
    t0 = p["t0"]
    sigma_alpha = p["sigmaAlpha"]
    sigma_beta = p["sigmaBeta"]
    sigma_tau = p["sigmaTau"]
    sigma_vec = np.array([sigma_alpha, sigma_beta, sigma_tau])
    nmc = p["Nmc"]
    seed = p["seed"]

    # Run simulation
    np.random.seed(seed)

    #y, t = thzgen(n, t, t0, 'taur', 0.4)
    y, t = thzgen(n, T, t0)
    sigma_t = sigmamu(sigma_vec, y, T)

    ym = np.tile(y, (500, 1)).T + np.tile(sigma_t, (nmc, 1)).T * np.random.rand(n, nmc)

    f = fftfreq(n, T)
    nf = int(n / 2) + 1
    ym_ft = np.fft.fft(ym, axis=0)
    ym_ratio = ym_ft[:, 0::2] / ym_ft[:, 1::2]

    vr = np.var(np.real(ym_ratio), axis=1)
    vi = np.var(np.imag(ym_ratio), axis=1)
    v = vr + vi

    data = {
        "t": t,
        "y0": y,
        "ym": ym,
        "P": p,
        "f": f,
        "Ym": ym_ft,
        "YmRatio": ym_ratio,
        "Vr": vr,
        "Vi": vi,
        "V": v,
        "Nf": nf,
        "sigma_t" : sigma_t
    }

    return data


