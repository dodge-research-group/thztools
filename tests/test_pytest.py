import numpy as np
import pytest

from thztools.thztools import (
    # costfunlsq,
    fftfreq,
    noisevar,
    # shiftmtx,
    # tdnll,
    # tdnoisefit,
    # tdtf,
    thzgen,
)

global fname


# ==================================================================
@pytest.fixture()
def arrangef():
    dt = 0.5
    f_odd_true = np.array([0.0, 0.4, 0.8, -0.8, -0.4])
    f_even_true = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -0.75, -0.5, -0.25])
    return f_odd_true, f_even_true, dt


@pytest.fixture()
def act(arrangef):
    f_odd = fftfreq(len(arrangef[0]), arrangef[2])
    f_even = fftfreq(len(arrangef[1]), arrangef[2])
    return f_odd, f_even


def test_fftfreq(arrangef, act):
    assert arrangef[0].all() == act[0].all()
    assert arrangef[1].all() == act[1].all()


# ============================================================================
# test noisevar
@pytest.fixture()
def arrange():
    sigma_alpha = (1e-4,)  # Additive noise amplitude [relative to peak]
    sigma_beta = (0.01,)  # Multiplicative noise amplitude [-]
    sigma_tau = (1e-3,)  # Time base noise amplitude [ps]
    np.array([sigma_alpha, sigma_beta, sigma_tau])

    y, t = thzgen(n=20, ts=0.05, t0=2.5)
    return y, t


@pytest.fixture()
def vmu(arrange):
    vmu = noisevar(arrange[0], arrange[1], 2.5)
    return vmu


def test_noisevar(vmu):
    vmu_true = np.array(
        [
            4.36242890663103e-6,
            1.18964333287937e-5,
            1.95293889001674e-5,
            2.55422643627414e-5,
            2.93161838588929e-5,
            3.09039003252619e-5,
            3.06822632940482e-5,
            2.90574034275636e-5,
            2.57635783418052e-5,
            1.82432209755600e-5,
            4.91505996620606e-6,
            2.38368994086406e-6,
            3.48676583001915e-5,
            8.17878674468416e-5,
            0.000100010020280983,
            8.17871774183767e-5,
            4.81773804791099e-05,
            1.93749781259091e-5,
            3.62858609090918e-6,
            1.05084791456656e-07,
        ]
    )

    assert vmu.all() == vmu_true.all()
