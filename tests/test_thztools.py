from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

if sys.version_info >= (3, 10):
    from typing import Concatenate
elif sys.version_info[:2] == (3, 9):
    from typing_extensions import Concatenate
else:
    msg = "Concatenate type annotation unavailable for Python 3.8 and earlier."
    raise ValueError(msg)

import thztools
from thztools.thztools import (
    NoiseModel,
    _assign_sampling_time,
    _costfuntls,
    _hess_noisefit,
    _jac_noisefit,
    _parse_noisefit_input,
    apply_frf,
    fit,
    get_option,
    noisefit,
    reset_option,
    scaleshift,
    set_option,
    wave,
)

eps = np.sqrt(np.finfo(np.float64).eps)
rtol = 1e-5
n_default = 16


def tfun(
    w: NDArray[np.float64], /, p0: float, p1: float
) -> NDArray[np.complex128]:
    return np.astype(p0 * np.exp(1j * p1 * w), np.complex128)


def jac_tfun(
    w: NDArray[np.float64], /, p0: float, p1: float
) -> NDArray[np.complex128]:
    return np.stack(
        (np.exp(1j * p1 * w), 1j * w * p0 * np.exp(1j * p1 * w)),
        dtype=np.complex128,
    ).T


def data_gen(n: int = n_default, *, np_sign: bool = True) -> dict[str, Any]:
    dt = 1.0 / n
    t = np.arange(n) * dt
    f = np.fft.rfftfreq(n, dt)
    x = np.sin(4 * pi * t)
    p0 = (0.5, dt)
    y = apply_frf(tfun, x, dt=dt, args=p0, numpy_sign_convention=np_sign)
    return {"dt": dt, "t": t, "f": f, "x": x, "p0": p0, "y": y}


# Reset options before each test
@pytest.fixture(autouse=True)
def global_reset() -> None:
    reset_option()


class TestOptions:
    def test_get_set_reset(self) -> None:
        assert get_option("sampling_time") is None
        set_option("sampling_time", 1.0)
        assert np.isclose(get_option("sampling_time"), 1.0)
        reset_option("sampling_time")
        assert get_option("sampling_time") is None

    @pytest.mark.parametrize("global_sampling_time", [None, 0.1])
    @pytest.mark.parametrize("dt", [None, 0.1, 1.0])
    def test_assignment(
        self, global_sampling_time: float | None, dt: float | None
    ) -> None:
        set_option("sampling_time", global_sampling_time)
        if global_sampling_time is None and dt is None:
            assert np.isclose(_assign_sampling_time(dt), 1.0)
        elif global_sampling_time is None and dt is not None:
            assert np.isclose(_assign_sampling_time(dt), dt)
        elif (
            global_sampling_time is not None
            and dt is None
            or (
                global_sampling_time is not None
                and dt is not None
                and np.isclose(dt, global_sampling_time)
            )
        ):
            assert np.isclose(_assign_sampling_time(dt), global_sampling_time)
        else:
            with pytest.warns(UserWarning):
                _assign_sampling_time(dt)


class TestNoiseModel:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    mu_dot = -2 * pi * np.sin(2 * pi * t)

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, np.ones(n)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, np.ones((2, n))),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, np.ones((n, 2))),
            (0, 1, 0, mu, dt, -1, mu**2),
            (0, 0, 1, mu, dt, -1, mu_dot**2),
            (0, 0, 1 / dt, mu, None, -1, mu_dot**2),
        ],
    )
    def test_var_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: NDArray[np.float64],
        dt: float | None,
        axis: int,
        expected: NDArray[np.float64],
    ) -> None:
        if dt is None:
            noise_model = NoiseModel(alpha, beta, tau)
            result = noise_model.noise_var(mu, axis=axis)
        else:
            noise_model = NoiseModel(alpha, beta, tau, dt=dt)
            result = noise_model.noise_var(mu, axis=axis)
        assert_allclose(result, expected, atol=eps, rtol=rtol)

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, np.ones(n)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, np.ones((2, n))),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, np.ones((n, 2))),
            (0, 1, 0, mu, dt, -1, np.abs(mu)),
            (0, 0, 1, mu, dt, -1, np.abs(mu_dot)),
            (0, 0, 1 / dt, mu, None, -1, np.abs(mu_dot)),
        ],
    )
    def test_amp_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: NDArray[np.float64],
        dt: float | None,
        axis: int,
        expected: NDArray[np.float64],
    ) -> None:
        if dt is None:
            noise_model = NoiseModel(alpha, beta, tau)
            result = noise_model.noise_amp(mu, axis=axis)
        else:
            noise_model = NoiseModel(alpha, beta, tau, dt=dt)
            result = noise_model.noise_amp(mu, axis=axis)
        assert_allclose(result, expected, atol=eps, rtol=rtol)

    @pytest.mark.parametrize(
        "alpha, beta, tau, mu, dt, axis, expected",
        [
            (1, 0, 0, mu, dt, -1, (n,)),
            (1, 0, 0, mu, None, -1, (n,)),
            (1, 0, 0, np.stack((mu, mu)), dt, -1, (2, n)),
            (1, 0, 0, np.stack((mu, mu)).T, dt, 0, (n, 2)),
        ],
    )
    def test_noise_definition(
        self,
        alpha: float,
        beta: float,
        tau: float,
        mu: ArrayLike,
        dt: float | None,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        if dt is None:
            noise_model = NoiseModel(alpha, beta, tau)
            result = noise_model.noise_sim(mu, axis=axis)
        else:
            noise_model = NoiseModel(alpha, beta, tau, dt=dt)
            result = noise_model.noise_sim(mu, axis=axis)
        assert result.shape == expected


class TestTransferOut:
    @pytest.mark.parametrize("fft_sign", [True, False])
    def test_inputs(self, *, fft_sign: bool) -> None:
        d = data_gen()
        dt = d["dt"]
        x = d["x"]
        expected = x
        assert_allclose(
            apply_frf(
                tfun, x, dt=dt, numpy_sign_convention=fft_sign, args=(1.0, 0.0)
            ),
            expected,
            atol=eps,
            rtol=rtol,
        )

    @pytest.mark.parametrize("x", [np.ones((n_default, n_default))])
    def test_error(self, x: NDArray[np.float64]) -> None:
        d = data_gen()
        dt = d["dt"]
        with pytest.raises(
            ValueError, match="x must be a one-dimensional array"
        ):
            _ = apply_frf(tfun, x, dt=dt, args=(1.0, 0.0))


class TestTimebase:
    @pytest.mark.parametrize(
        "dt",
        [
            None,
            1.0,
            2.0,
        ],
    )
    @pytest.mark.parametrize(
        "t_init",
        [
            None,
            1.0,
            2.0,
        ],
    )
    def test_timebase(self, dt: float | None, t_init: float | None) -> None:
        n = 8
        if t_init is None:
            t = thztools.timebase(n, dt=dt)
            t_init = 0.0
        else:
            t = thztools.timebase(n, dt=dt, t_init=t_init)
        dt = _assign_sampling_time(dt)
        t_expected = t_init + np.arange(n) * dt
        assert_allclose(t, t_expected, rtol=rtol, atol=eps)


class TestWave:
    @pytest.mark.parametrize(
        "t0",
        [2.4, None],
    )
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"a": 1.0},
            {"taur": 6.0},
            {"tauc": 2.0},
            {"fwhm": 1.0},
        ],
    )
    def test_inputs(self, t0: float | None, kwargs: dict[str, Any]) -> None:
        n = 8
        dt = 1.0
        y_expected = np.array(
            [
                0.07767792,
                -0.63041598,
                -1.03807638,
                -0.81199085,
                -0.03955531,
                0.72418661,
                1.0,
                0.71817398,
            ]
        )
        assert_allclose(
            wave(n, dt=dt, t0=t0, **kwargs),
            y_expected,
            atol=eps,
            rtol=rtol,
        )


class TestScaleShift:
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    x = np.cos(2 * pi * t)
    x_2 = np.stack((x, x))

    @pytest.mark.parametrize(
        "x, kwargs, expected",
        [
            ([], {}, np.empty((0,))),
            (x, {}, x),
            (x, {"a": 2}, 2 * x),
            (x, {"eta": 1}, np.cos(2 * pi * (t - dt))),
            (x, {"a": 2, "eta": 1}, 2 * np.cos(2 * pi * (t - dt))),
            (x, {"a": 2, "eta": dt, "dt": dt}, 2 * np.cos(2 * pi * (t - dt))),
            (x_2, {"a": [2, 0.5]}, np.stack((2 * x, 0.5 * x))),
            (
                x_2,
                {"eta": [1, -1]},
                np.stack(
                    (np.cos(2 * pi * (t - dt)), np.cos(2 * pi * (t + dt)))
                ),
            ),
            (
                x_2,
                {"eta": [dt, -dt], "dt": dt},
                np.stack(
                    (np.cos(2 * pi * (t - dt)), np.cos(2 * pi * (t + dt)))
                ),
            ),
            (
                x_2,
                {"a": [2, 0.5], "eta": [1, -1]},
                np.stack(
                    (
                        2 * np.cos(2 * pi * (t - dt)),
                        0.5 * np.cos(2 * pi * (t + dt)),
                    )
                ),
            ),
            (
                x_2,
                {"a": [2, 0.5], "eta": [dt, -dt], "dt": dt},
                np.stack(
                    (
                        2 * np.cos(2 * pi * (t - dt)),
                        0.5 * np.cos(2 * pi * (t + dt)),
                    )
                ),
            ),
            (x_2.T, {"a": [2, 0.5], "axis": 0}, np.stack((2 * x, 0.5 * x)).T),
        ],
    )
    def test_inputs(
        self,
        x: NDArray[np.float64],
        kwargs: dict[str, Any],
        expected: NDArray[np.float64],
    ) -> None:
        assert_allclose(
            scaleshift(x, **kwargs),
            expected,
            atol=eps,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "x, kwargs", [(x, {"a": [2, 0.5]}), (x, {"eta": [1, -1]})]
    )
    def test_errors(self, x: ArrayLike, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="correction with shape"):
            scaleshift(x, **kwargs)


class TestCostFunTLS:
    def test_output(self) -> None:
        theta = (1, 0)
        mu = np.arange(8)
        xx = mu
        yy = xx
        sigmax = np.ones_like(xx)
        sigmay = sigmax
        dt = 1.0

        assert_allclose(
            _costfuntls(
                tfun,
                theta,
                mu,
                xx,
                yy,
                sigmax,
                sigmay,
                dt,
            ),
            np.concatenate((np.zeros_like(xx), np.zeros_like(xx))),
            atol=eps,
            rtol=rtol,
        )


class TestJacNoiseFit:
    m = 2
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    x = np.tile(mu, [m, 1])
    logv_alpha = 0
    logv_beta = -np.inf
    logv_tau = -np.inf
    delta_mu = np.zeros(n)
    delta_a = np.zeros(m - 1)
    eta = np.zeros(m - 1)
    desired_nll = x.size * np.log(2 * pi) / 2

    @pytest.mark.parametrize(
        "fix_logv_alpha, desired_gradnll_logv_alpha",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.array([m * n / 2]),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_logv_beta, desired_gradnll_logv_beta",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.array([0.0]),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_logv_tau, desired_gradnll_logv_tau",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.array([0.0]),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_delta_mu, desired_gradnll_delta_mu",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.zeros(n),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_delta_a, desired_gradnll_delta_a",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.zeros(m - 1),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_eta, desired_gradnll_eta",
        [
            (
                True,
                np.array([]),
            ),
            (
                False,
                np.zeros(m - 1),
            ),
        ],
    )
    def test_gradnll_calc(
        self,
        *,
        fix_logv_alpha: bool,
        fix_logv_beta: bool,
        fix_logv_tau: bool,
        fix_delta_mu: bool,
        fix_delta_a: bool,
        fix_eta: bool,
        desired_gradnll_logv_alpha: NDArray[np.float64],
        desired_gradnll_logv_beta: NDArray[np.float64],
        desired_gradnll_logv_tau: NDArray[np.float64],
        desired_gradnll_delta_mu: NDArray[np.float64],
        desired_gradnll_delta_a: NDArray[np.float64],
        desired_gradnll_eta: NDArray[np.float64],
    ) -> None:
        n = self.n
        m = self.m
        x = self.x
        logv_alpha = self.logv_alpha
        logv_beta = self.logv_beta
        logv_tau = self.logv_tau
        delta_mu = self.delta_mu
        delta_a = self.delta_a
        eta_on_dt = np.asarray(self.eta / self.dt, np.float64)
        desired_gradnll = np.concatenate(
            (
                desired_gradnll_logv_alpha,
                desired_gradnll_logv_beta,
                desired_gradnll_logv_tau,
                desired_gradnll_delta_mu,
                desired_gradnll_delta_a,
                desired_gradnll_eta,
            )
        )
        gradnll = _jac_noisefit(
            x,
            logv_alpha,
            logv_beta,
            logv_tau,
            delta_mu,
            delta_a,
            eta_on_dt,
            fix_logv_alpha=fix_logv_alpha,
            fix_logv_beta=fix_logv_beta,
            fix_logv_tau=fix_logv_tau,
            fix_delta_mu=fix_delta_mu,
            fix_delta_a=fix_delta_a,
            fix_eta=fix_eta,
            scale_logv_alpha=1.0,
            scale_logv_beta=1.0,
            scale_logv_tau=1.0,
            scale_delta_mu=np.ones(n),
            scale_delta_a=np.ones(m - 1),
            scale_eta_on_dt=np.ones(m - 1),
        )
        assert_allclose(gradnll, desired_gradnll, atol=10 * eps, rtol=rtol)


class TestHessNoiseFit:
    m = 2
    n = 4
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)
    x = np.tile(mu, [m, 1])
    logv_alpha = 0
    logv_beta = 0
    logv_tau = 0
    delta_mu = np.zeros(n)
    delta_a = np.zeros(m - 1)
    eta = np.zeros(m - 1)
    scale_logv_alpha = 1.0
    scale_logv_beta = 1.0
    scale_logv_tau = 1.0
    scale_delta_mu = np.ones_like(delta_mu)
    scale_delta_a = np.ones_like(delta_a)
    scale_eta = np.ones_like(eta)

    def test_hess_logv_logv(self) -> None:
        desired_hess_logv_logv = np.array(
            [
                [
                    9.1045125168940400e-01,
                    -5.0000000000000000e-01,
                    -4.1045125168940388e-01,
                ],
                [
                    -5.0000000000000000e-01,
                    5.0000000000000000e-01,
                    -2.6197273189757708e-32,
                ],
                [
                    -4.1045125168940388e-01,
                    -2.6197273189757708e-32,
                    4.1045125168940433e-01,
                ],
            ]
        )
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_logv_logv = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=False,
            fix_logv_beta=False,
            fix_logv_tau=False,
            fix_delta_mu=True,
            fix_delta_a=True,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )
        assert_allclose(
            hess_logv_logv, desired_hess_logv_logv, atol=eps, rtol=rtol
        )

    def test_hess_logv_delta_mu(self) -> None:
        desired_hess_logv_mu = np.array(
            [
                [
                    9.1045125168940388e-01,
                    1.6127071987048048e-16,
                    -9.1045125168940388e-01,
                    -1.8164267364532367e-16,
                ],
                [
                    -5.0000000000000000e-01,
                    1.1576587551626463e-16,
                    5.0000000000000000e-01,
                    -4.5128140582676141e-17,
                ],
                [
                    -4.1045125168940416e-01,
                    -2.7703659538674511e-16,
                    4.1045125168940416e-01,
                    2.2677081422799980e-16,
                ],
            ]
        )
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_logv_mu = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=False,
            fix_logv_beta=False,
            fix_logv_tau=False,
            fix_delta_mu=False,
            fix_delta_a=True,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[:3, 3:]
        assert_allclose(
            hess_logv_mu, desired_hess_logv_mu, atol=eps, rtol=rtol
        )

    def test_hess_logv_delta_a(self) -> None:
        desired_hess_logv_a = np.array(
            [[-0.9104512516894039], [0.5], [0.4104512516894041]]
        )
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_logv_a = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=False,
            fix_logv_beta=False,
            fix_logv_tau=False,
            fix_delta_mu=True,
            fix_delta_a=False,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[:3, 3:]
        assert_allclose(hess_logv_a, desired_hess_logv_a, atol=eps, rtol=rtol)

    def test_hess_logv_eta(self) -> None:
        desired_hess_logv_eta = np.array(
            [
                [-3.767308415119052e-16],
                [-8.901975977273483e-16],
                [1.266928439239253e-15],
            ]
        )
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_logv_eta = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=False,
            fix_logv_beta=False,
            fix_logv_tau=False,
            fix_delta_mu=True,
            fix_delta_a=True,
            fix_eta=False,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[:3, 3:]
        assert_allclose(
            hess_logv_eta, desired_hess_logv_eta, atol=eps, rtol=rtol
        )

    def test_hess_delta_mu_delta_mu(self) -> None:
        desired_hess_mu_mu = np.array(
            [
                [
                    6.9885169083140497e-01,
                    -1.7621763356243157e-16,
                    3.0114830916859503e-01,
                    2.2648341472117688e-16,
                ],
                [
                    -1.7621763356243157e-16,
                    2.3873023067041736e00,
                    1.7621763356243157e-16,
                    -1.2337005501361697e00,
                ],
                [
                    3.0114830916859503e-01,
                    1.7621763356243157e-16,
                    6.9885169083140497e-01,
                    -2.2648341472117688e-16,
                ],
                [
                    2.2648341472117688e-16,
                    -1.2337005501361697e00,
                    -2.2648341472117688e-16,
                    2.3873023067041736e00,
                ],
            ]
        )
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_mu_mu = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=False,
            fix_delta_a=True,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )
        assert_allclose(hess_mu_mu, desired_hess_mu_mu, atol=eps, rtol=rtol)

    def test_hess_delta_mu_delta_a(self) -> None:
        n = self.n
        desired_hess_mu_a = [
            [-1.4104512516894041e00],
            [-1.7893015360387762e-16],
            [1.4104512516894041e00],
            [2.3462097484551508e-16],
        ]
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_mu_a = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=False,
            fix_delta_a=False,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[:n, n:]
        assert_allclose(hess_mu_a, desired_hess_mu_a, atol=eps, rtol=rtol)

    def test_hess_delta_mu_eta(self) -> None:
        n = self.n
        desired_hess_mu_eta = [
            [1.0890338318440097e-16],
            [-3.7630114147090579e00],
            [-3.8494520984901552e-16],
            [3.7630114147090579e00],
        ]
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_mu_eta = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=False,
            fix_delta_a=True,
            fix_eta=False,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[:n, n:]
        assert_allclose(hess_mu_eta, desired_hess_mu_eta, atol=eps, rtol=rtol)

    def test_hess_delta_a_delta_a(self) -> None:
        desired_hess_a_a = [[0.3977033816628099]]
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_a_a = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=True,
            fix_delta_a=False,
            fix_eta=True,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )
        assert_allclose(hess_a_a, desired_hess_a_a, atol=eps, rtol=rtol)

    def test_hess_delta_a_eta(self) -> None:
        m = self.m
        desired_hess_a_eta = [[4.278233838022636e-16]]
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_a_eta = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=True,
            fix_delta_a=False,
            fix_eta=False,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )[: m - 1, m - 1 :]
        assert_allclose(hess_a_eta, desired_hess_a_eta, atol=eps, rtol=rtol)

    def test_hess_eta_eta(self) -> None:
        desired_hess_eta_eta = [[47.28739606329804]]
        scale_eta_on_dt = np.asarray(self.scale_eta / self.dt, np.float64)
        hess_eta_eta = _hess_noisefit(
            self.x,
            self.logv_alpha,
            self.logv_beta,
            self.logv_tau,
            self.delta_mu / self.scale_delta_mu,
            self.delta_a / self.scale_delta_a,
            self.eta / self.scale_eta,
            fix_logv_alpha=True,
            fix_logv_beta=True,
            fix_logv_tau=True,
            fix_delta_mu=True,
            fix_delta_a=True,
            fix_eta=False,
            scale_logv_alpha=self.scale_logv_alpha,
            scale_logv_beta=self.scale_logv_beta,
            scale_logv_tau=self.scale_logv_tau,
            scale_delta_mu=self.scale_delta_mu,
            scale_delta_a=self.scale_delta_a,
            scale_eta_on_dt=scale_eta_on_dt,
        )
        assert_allclose(
            hess_eta_eta, desired_hess_eta_eta, atol=eps, rtol=rtol
        )


class TestNoiseFit:
    rng = np.random.default_rng(0)
    n = 256
    m = 64
    dt = 0.05
    t = np.arange(n) * dt
    mu = wave(n, dt=dt, t0=n * dt / 3)
    alpha, beta, tau = 1e-5, 1e-3, 1e-3
    sigma = np.array([alpha, beta, tau])
    noise_model = NoiseModel(alpha, beta, tau, dt=dt)
    noise = noise_model.noise_sim(np.ones((m, 1)) * mu, seed=0)
    noise_amp = noise_model.noise_amp(mu)
    x = np.array(mu + noise)
    a = np.ones(m)
    eta = np.zeros(m)
    scale_delta_a = 1e-2 * np.ones(m - 1)
    scale_eta = 1e-3 * np.ones(m - 1) / dt

    @pytest.mark.parametrize(
        "x, mu0, a0, eta0, fix_sigma_alpha, fix_sigma_beta, fix_sigma_tau, "
        "fix_mu, fix_a, fix_eta, pattern",
        [
            (
                x[:, 0],
                mu,
                a,
                eta,
                False,
                False,
                False,
                False,
                False,
                False,
                "Data array x must be 2D",
            ),
            (
                x,
                [],
                a,
                eta,
                False,
                False,
                False,
                False,
                False,
                False,
                "Size of mu0 is incompatible with data array x",
            ),
            (
                x,
                mu,
                [],
                eta,
                False,
                False,
                False,
                False,
                False,
                False,
                "Size of a0 is incompatible with data array x",
            ),
            (
                x,
                mu,
                a,
                [],
                False,
                False,
                False,
                False,
                False,
                False,
                "Size of eta0 is incompatible with data array x",
            ),
            (
                x,
                mu,
                a,
                eta,
                True,
                True,
                True,
                True,
                True,
                True,
                "All variables are fixed",
            ),
        ],
    )
    def test_exceptions(
        self,
        x: NDArray[np.float64],
        mu0: NDArray[np.float64],
        a0: NDArray[np.float64],
        eta0: NDArray[np.float64],
        *,
        fix_sigma_alpha: bool,
        fix_sigma_beta: bool,
        fix_sigma_tau: bool,
        fix_mu: bool,
        fix_a: bool,
        fix_eta: bool,
        pattern: str,
    ) -> None:
        m = self.m
        n = self.n
        dt = self.dt
        sigma_alpha0 = self.alpha
        sigma_beta0 = self.beta
        sigma_tau0 = self.tau
        with pytest.raises(ValueError, match=pattern):
            _ = _parse_noisefit_input(
                x.T,
                dt=dt,
                sigma_alpha0=sigma_alpha0,
                sigma_beta0=sigma_beta0,
                sigma_tau0=sigma_tau0,
                mu0=mu0,
                a0=a0,
                eta0=eta0,
                fix_sigma_alpha=fix_sigma_alpha,
                fix_sigma_beta=fix_sigma_beta,
                fix_sigma_tau=fix_sigma_tau,
                fix_mu=fix_mu,
                fix_a=fix_a,
                fix_eta=fix_eta,
                scale_logv_alpha=1.0,
                scale_logv_beta=1.0,
                scale_logv_tau=1.0,
                scale_delta_mu=np.ones(n),
                scale_delta_a=np.ones(m - 1),
                scale_eta=np.ones(m - 1),
            )

    @pytest.mark.parametrize("sigma_alpha0", [None, alpha, 0.0])
    @pytest.mark.parametrize("sigma_beta0", [None, beta, 0.0])
    @pytest.mark.parametrize("sigma_tau0", [None, tau, 0.0])
    @pytest.mark.parametrize("mu0", [None, mu])
    @pytest.mark.parametrize("a0", [None, a])
    @pytest.mark.parametrize("eta0", [None, eta])
    @pytest.mark.parametrize(
        "scale_sigma_alpha, scale_sigma_beta, scale_sigma_tau, "
        "scale_delta_mu, scale_delta_a, scale_eta",
        [
            (None, None, None, None, None, None),
            (alpha, beta, tau / dt, noise_amp, scale_delta_a, scale_eta),
        ],
    )
    def test_inputs(
        self,
        sigma_alpha0: float | None,
        sigma_beta0: float | None,
        sigma_tau0: float | None,
        mu0: ArrayLike | None,
        a0: ArrayLike | None,
        eta0: ArrayLike | None,
        scale_sigma_alpha: float | None,
        scale_sigma_beta: float | None,
        scale_sigma_tau: float | None,
        scale_delta_mu: ArrayLike | None,
        scale_delta_a: ArrayLike | None,
        scale_eta: ArrayLike | None,
    ) -> None:
        x = self.x
        dt = self.dt
        _ = _parse_noisefit_input(
            x.T,
            dt=dt,
            sigma_alpha0=sigma_alpha0,
            sigma_beta0=sigma_beta0,
            sigma_tau0=sigma_tau0,
            mu0=mu0,
            a0=a0,
            eta0=eta0,
            fix_sigma_alpha=False,
            fix_sigma_beta=False,
            fix_sigma_tau=False,
            fix_mu=False,
            fix_a=False,
            fix_eta=False,
            scale_logv_alpha=scale_sigma_alpha,
            scale_logv_beta=scale_sigma_beta,
            scale_logv_tau=scale_sigma_tau,
            scale_delta_mu=scale_delta_mu,
            scale_delta_a=scale_delta_a,
            scale_eta=scale_eta,
        )

    @pytest.mark.parametrize(
        "fix_sigma_alpha, fix_sigma_beta, fix_sigma_tau, fix_mu, fix_a, "
        "fix_eta",
        [
            (False, False, False, False, False, False),
            (True, False, False, False, False, False),
            (False, True, False, False, False, False),
            (False, False, True, False, False, False),
            (False, False, False, True, False, False),
            (False, False, False, False, True, False),
            (False, False, False, False, False, True),
        ],
    )
    def test_noisefit(
        self,
        *,
        fix_sigma_alpha: bool,
        fix_sigma_beta: bool,
        fix_sigma_tau: bool,
        fix_mu: bool,
        fix_a: bool,
        fix_eta: bool,
    ) -> None:
        x = self.x
        dt = self.dt
        sigma_alpha0 = self.alpha
        sigma_beta0 = self.beta
        sigma_tau0 = self.tau
        mu0 = self.mu
        a0 = self.a
        eta0 = self.eta
        result = noisefit(
            x.T,
            dt=dt,
            sigma_alpha0=sigma_alpha0,
            sigma_beta0=sigma_beta0,
            sigma_tau0=sigma_tau0,
            mu0=mu0,
            a0=a0,
            eta0=eta0,
            fix_sigma_alpha=fix_sigma_alpha,
            fix_sigma_beta=fix_sigma_beta,
            fix_sigma_tau=fix_sigma_tau,
            fix_mu=fix_mu,
            fix_a=fix_a,
            fix_eta=fix_eta,
        )
        assert result.diagnostic["status"] == 0

        sigma = self.sigma
        sigma_est = np.asarray(
            [
                result.noise_model.sigma_alpha,
                result.noise_model.sigma_beta,
                result.noise_model.sigma_tau,
            ]
        )
        assert_allclose(sigma_est / sigma, np.ones(3), atol=1e-1, rtol=1e-1)


class TestFit:
    alpha, beta, tau = 1e-5, 0, 0
    sigma = np.array([alpha, beta, tau])

    @pytest.mark.parametrize("n", [15, 16])
    @pytest.mark.parametrize("numpy_sign_convention", [True, False])
    @pytest.mark.parametrize("noise_parms", [(1, 0, 0), sigma**2])
    @pytest.mark.parametrize(
        "f_bounds",
        [
            None,
            (-np.inf, np.inf),
            (-np.inf, -2),
            (-np.inf, -3),
            (1, np.inf),
            (1, -2),
            (1, -3),
            (2, np.inf),
            (2, -2),
            (2, -3),
        ],
    )
    @pytest.mark.parametrize("p_bounds", [None, ((0, -1), (2, 1))])
    @pytest.mark.parametrize("jac", [None, jac_tfun])
    @pytest.mark.parametrize("lsq_options", [None, {}, {"verbose": 0}])
    def test_inputs(
        self,
        n: int,
        *,
        numpy_sign_convention: bool,
        noise_parms: ArrayLike,
        f_bounds: tuple[Any, Any] | None,
        p_bounds: ArrayLike | None,
        jac: Callable[
            Concatenate[NDArray[np.float64], ...], NDArray[np.complex128]
        ],
        lsq_options: dict[str, Any],
    ) -> None:
        d = data_gen(n, np_sign=numpy_sign_convention)
        dt = d["dt"]
        f = d["f"]
        x = d["x"]
        y = d["y"]
        p0 = d["p0"]

        if f_bounds is not None:
            # Check if either the upper or lower bound is an integer. If it is,
            # replace it with the value of the frequency array f at that index
            # value. Need to change from tuple to list and back to modify
            # individual elements.
            f_list = list(f_bounds)
            for i in range(2):
                f_list[i] = (
                    f[f_list[i]] if isinstance(f_list[i], int) else f_list[i]
                )
            f_bounds = tuple(f_list)

        p = fit(
            tfun,
            x,
            y,
            p0,
            noise_parms,
            dt=dt,
            numpy_sign_convention=numpy_sign_convention,
            f_bounds=f_bounds,
            p_bounds=p_bounds,
            jac=jac,
            lsq_options=lsq_options,
        )
        assert_allclose(p.p_opt, p0)

    @pytest.mark.parametrize("numpy_sign_convention", [True, False])
    @pytest.mark.parametrize("jac", [None, jac_tfun])
    def test_args(
        self,
        *,
        numpy_sign_convention: bool,
        jac: Callable[
            Concatenate[NDArray[np.float64], ...], NDArray[np.complex128]
        ]
        | None,
    ) -> None:
        sigma = self.sigma
        d = data_gen(np_sign=numpy_sign_convention)
        p0 = d["p0"]
        x = d["x"]
        dt = d["dt"]
        y = d["y"]
        p = fit(
            tfun,
            x,
            y,
            (p0[0],),
            noise_parms=sigma,
            dt=dt,
            numpy_sign_convention=numpy_sign_convention,
            args=(p0[1],),
            jac=jac,
        )
        assert_allclose(p.p_opt, p0[0])

    @pytest.mark.parametrize("numpy_sign_convention", [True, False])
    @pytest.mark.parametrize("jac", [None, jac_tfun])
    def test_kwargs(
        self,
        *,
        numpy_sign_convention: bool,
        jac: Callable[
            Concatenate[NDArray[np.float64], ...], NDArray[np.complex128]
        ]
        | None,
    ) -> None:
        sigma = self.sigma
        d = data_gen(np_sign=numpy_sign_convention)
        p0 = d["p0"]
        x = d["x"]
        dt = d["dt"]
        y = d["y"]
        p = fit(
            tfun,
            x,
            y,
            (p0[0]),
            noise_parms=sigma,
            dt=dt,
            numpy_sign_convention=numpy_sign_convention,
            kwargs={"p1": p0[1]},
            jac=jac,
        )
        assert_allclose(p.p_opt, p0[0])

    def test_errors(self) -> None:
        d = data_gen()
        p0 = d["p0"]
        x = d["x"]
        dt = d["dt"]
        y = d["y"]

        with pytest.raises(
            ValueError, match="`bounds` must contain 2 elements."
        ):
            _ = fit(tfun, x, y, p0, dt=dt, p_bounds=())

        with pytest.raises(
            ValueError, match="sigma_parms must be a tuple of length"
        ):
            _ = fit(tfun, x, y, p0, (0, 0), dt=dt)

        with pytest.raises(KeyError, match="Invalid key"):
            _ = fit(tfun, x, y, p0, dt=dt, lsq_options={"bad_key": 0})
