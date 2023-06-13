import numpy as np
from numpy import pi
from numpy.testing import assert_array_almost_equal
import pytest

from thztools.thztools import (
    # costfunlsq,
    fftfreq,
    noisevar,
    noiseamp,
    # shiftmtx,
    # tdnll,
    # tdnoisefit,
    # tdtf,
    thzgen,
)


class TestFFTFreq:
    @pytest.mark.parametrize(
        "n, ts, expected",
        [
            (9, 1, [0, 1, 2, 3, 4, -4, -3, -2, -1]),
            (9, pi, [0, 1, 2, 3, 4, -4, -3, -2, -1]),
            (10, 1, [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]),
            (10, pi, [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]),
        ],
    )
    def test_definition(self, n, ts, expected):
        assert_array_almost_equal(n * ts * fftfreq(n, ts), expected)


class TestNoise:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    mu_dot = -2 * pi * np.sin(2 * pi * t)

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [([1, 0, 0], mu, dt, np.ones(n)),
         ([0, 1, 0], mu, dt, mu ** 2),
         ([0, 0, 1], mu, dt, mu_dot ** 2)]
    )
    def test_var_definition(self, sigma, mu, dt, expected):
        assert_array_almost_equal(noisevar(sigma, mu, dt), expected)

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [([1, 0, 0], mu, dt, np.ones(n)),
         ([0, 1, 0], mu, dt, np.abs(mu)),
         ([0, 0, 1], mu, dt, np.abs(mu_dot))]
    )
    def test_amp_definition(self, sigma, mu, dt, expected):
        assert_array_almost_equal(noiseamp(sigma, mu, dt), expected)


class TestTHzGen:

    @pytest.mark.parametrize(
        "kwargs",
        [dict(),
         dict(a=1.0),
         dict(taur=0.3),
         dict(tauc=0.1),
         dict(fwhm=0.05),
         ]
    )
    @pytest.mark.parametrize(
        "n, ts, t0, expected",
        [(8, 1, 2,
          ([0.05651348, -0.13522073, 1., -0.65804181, -0.28067975,
           0.05182924, -0.01837401, -0.01602642],
           np.arange(8)),
          )]
    )
    def test_definition(self, n, ts, t0, kwargs, expected):
        assert_array_almost_equal(thzgen(n, ts, t0, **kwargs),
                                  expected)
