import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike

from thztools.thztools import (
    # costfunlsq,
    noiseamp,
    noisevar,
    scaleshift,
    # tdnll,
    # tdnoisefit,
    thzgen,
)

atol = 1e-8
rtol = 1e-5


class TestNoise:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    mu_dot = -2 * pi * np.sin(2 * pi * t)

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [
            ([1, 0, 0], mu, dt, np.ones(n)),
            ([0, 1, 0], mu, dt, mu**2),
            ([0, 0, 1], mu, dt, mu_dot**2),
        ],
    )
    def test_var_definition(self, sigma: ArrayLike, mu: ArrayLike, dt: float,
                            expected: ArrayLike) -> None:
        assert_allclose(
            noisevar(sigma, mu, dt), expected, atol=atol, rtol=rtol
        )

    @pytest.mark.parametrize(
        "sigma, mu, dt, expected",
        [
            ([1, 0, 0], mu, dt, np.ones(n)),
            ([0, 1, 0], mu, dt, np.abs(mu)),
            ([0, 0, 1], mu, dt, np.abs(mu_dot)),
        ],
    )
    def test_amp_definition(self, sigma: ArrayLike, mu: ArrayLike, dt: float,
                            expected: ArrayLike) -> None:
        assert_allclose(
            noiseamp(sigma, mu, dt), expected, atol=atol, rtol=rtol
        )


class TestTHzGen:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"a": 1.0},
            {"taur": 0.3},
            {"tauc": 0.1},
            {"fwhm": 0.05},
        ],
    )
    def test_inputs(self, kwargs: dict) -> None:
        n = 8
        ts = 1.0
        t0 = 2.0
        y_expected = np.array(
            [
                0.05651348,
                -0.13522073,
                1.0,
                -0.65804181,
                -0.28067975,
                0.05182924,
                -0.01837401,
                -0.01602642,
            ]
        )
        t_expected = np.arange(n)
        assert_allclose(
            thzgen(n, ts, t0, **kwargs),
            (y_expected, t_expected),
            atol=atol,
            rtol=rtol,
        )


class TestScaleShift:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    x = np.cos(2 * pi * t)

    @pytest.mark.parametrize(
        "x, kwargs, expected",
        [
            [x, {}, x],
            [x, {"a": 2}, 2 * x],
            [x, {"eta": 1}, np.cos(2 * pi * (t - dt))],
            [x, {"a": 2, "eta": 1}, 2 * np.cos(2 * pi * (t - dt))],
            [x, {"a": 2, "eta": dt, "ts": dt}, 2 * np.cos(2 * pi * (t - dt))],
        ],
    )
    def test_inputs(
        self, x: ArrayLike, kwargs: dict, expected: ArrayLike
    ) -> None:
        assert_allclose(
            scaleshift(x, **kwargs),
            expected,
            atol=atol,
            rtol=rtol,
        )
