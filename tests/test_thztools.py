from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy import pi
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

import thztools
from thztools.thztools import (
    NoiseModel,
    _assign_sampling_time,
    _costfuntls,
    _hess_noisefit,
    _jac_noisefit,
    _parse_noisefit_input,
    fit,
    get_option,
    noisefit,
    reset_option,
    scaleshift,
    set_option,
    transfer,
    wave,
)

eps = np.sqrt(np.finfo(np.float64).eps)
rtol = 1e-5


def tfun(w, p0, p1):
    return p0 * np.exp(1j * p1 * w)


def jac_fun(w, p):
    exp_ipw = np.exp(1j * p[1] * w)
    return np.stack((exp_ipw, 1j * w * p[0] * exp_ipw)).T


# Reset options before each test
@pytest.fixture(autouse=True)
def global_reset():
    reset_option()


class TestOptions:
    def test_get_set_reset(self):
        assert get_option("sampling_time") is None
        set_option("sampling_time", 1.0)
        assert np.isclose(get_option("sampling_time"), 1.0)
        reset_option("sampling_time")
        assert get_option("sampling_time") is None

    @pytest.mark.parametrize("global_sampling_time", [None, 0.1])
    @pytest.mark.parametrize("dt", [None, 0.1, 1.0])
    def test_assignment(self, global_sampling_time, dt):
        set_option("sampling_time", global_sampling_time)
        if global_sampling_time is None and dt is None:
            assert np.isclose(_assign_sampling_time(dt), 1.0)
        elif global_sampling_time is None and dt is not None:
            assert np.isclose(_assign_sampling_time(dt), dt)
        elif (
            global_sampling_time is not None
            and dt is None
            or global_sampling_time is not None
            and np.isclose(dt, global_sampling_time)
        ):
            assert np.isclose(_assign_sampling_time(dt), global_sampling_time)
        else:
            with pytest.warns(UserWarning):
                assert np.isclose(_assign_sampling_time(dt), dt)


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
        mu: ArrayLike,
        dt: float | None,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        if dt is None:
            noise_model = NoiseModel(alpha, beta, tau)
            result = noise_model.noise_var(mu, axis=axis)
        else:
            noise_model = NoiseModel(alpha, beta, tau, dt=dt)
            result = noise_model.noise_var(mu, axis=axis)
        assert_allclose(result, expected, atol=eps, rtol=rtol)  # type: ignore

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
        mu: ArrayLike,
        dt: float,
        axis: int,
        expected: ArrayLike,
    ) -> None:
        if dt is None:
            noise_model = NoiseModel(alpha, beta, tau)
            result = noise_model.noise_amp(mu, axis=axis)
        else:
            noise_model = NoiseModel(alpha, beta, tau, dt=dt)
            result = noise_model.noise_amp(mu, axis=axis)
        assert_allclose(result, expected, atol=eps, rtol=rtol)  # type: ignore

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
        dt: float,
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
    n = 16
    dt = 1.0 / n
    t = np.arange(n) * dt
    mu = np.cos(2 * pi * t)

    @pytest.mark.parametrize("fft_sign", [True, False])
    def test_inputs(self, fft_sign):
        ts = self.dt
        x = self.mu
        expected = x
        assert_allclose(
            transfer(
                tfun, x, dt=ts, numpy_sign_convention=fft_sign, args=(1.0, 0.0)
            ),
            expected,
            atol=eps,
            rtol=rtol,
        )

    @pytest.mark.parametrize("x", [np.ones((n, n))])
    def test_error(self, x):
        dt = self.dt
        with pytest.raises(
            ValueError, match="x must be a one-dimensional array"
        ):
            _ = transfer(tfun, x, dt=dt, args=(1.0, 0.0))


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
    def test_timebase(self, dt, t_init):
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
    def test_inputs(self, t0: float | None, kwargs: dict) -> None:
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
            wave(n, dt=dt, t0=t0, **kwargs),  # type: ignore
            y_expected,  # type: ignore
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
        self, x: ArrayLike, kwargs: dict, expected: ArrayLike
    ) -> None:
        assert_allclose(
            scaleshift(x, **kwargs),  # type: ignore
            expected,  # type: ignore
            atol=eps,
            rtol=rtol,
        )

    @pytest.mark.parametrize(
        "x, kwargs", [(x, {"a": [2, 0.5]}), (x, {"eta": [1, -1]})]
    )
    def test_errors(self, x: ArrayLike, kwargs: dict) -> None:
        with pytest.raises(ValueError, match="correction with shape"):
            scaleshift(x, **kwargs)


class TestCostFunTLS:
    theta = (1, 0)
    mu = np.arange(8)
    xx = mu
    yy = xx
    sigmax = np.ones_like(xx)
    sigmay = sigmax
    w_below_idx = np.zeros(mu.size, dtype=bool)
    w_in_idx = np.ones(mu.size, dtype=bool)
    w_above_idx = np.zeros(mu.size, dtype=bool)
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
            # w_below_idx,
            # w_in_idx,
            # w_above_idx,
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
                [],
            ),
            (
                False,
                [m * n / 2],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_logv_beta, desired_gradnll_logv_beta",
        [
            (
                True,
                [],
            ),
            (
                False,
                [0.0],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_logv_tau, desired_gradnll_logv_tau",
        [
            (
                True,
                [],
            ),
            (
                False,
                [0.0],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "fix_delta_mu, desired_gradnll_delta_mu",
        [
            (
                True,
                [],
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
                [],
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
                [],
            ),
            (
                False,
                np.zeros(m - 1),
            ),
        ],
    )
    def test_gradnll_calc(
        self,
        fix_logv_alpha,
        fix_logv_beta,
        fix_logv_tau,
        fix_delta_mu,
        fix_delta_a,
        fix_eta,
        desired_gradnll_logv_alpha,
        desired_gradnll_logv_beta,
        desired_gradnll_logv_tau,
        desired_gradnll_delta_mu,
        desired_gradnll_delta_a,
        desired_gradnll_eta,
    ):
        n = self.n
        m = self.m
        x = self.x
        logv_alpha = self.logv_alpha
        logv_beta = self.logv_beta
        logv_tau = self.logv_tau
        delta_mu = self.delta_mu
        delta_a = self.delta_a
        eta_on_dt = self.eta / self.dt
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

    def test_hess_logv_logv(self):
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )
        assert_allclose(
            hess_logv_logv, desired_hess_logv_logv, atol=eps, rtol=rtol
        )

    def test_hess_logv_delta_mu(self):
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[:3, 3:]
        assert_allclose(
            hess_logv_mu, desired_hess_logv_mu, atol=eps, rtol=rtol
        )

    def test_hess_logv_delta_a(self):
        desired_hess_logv_a = np.array(
            [[-0.9104512516894039], [0.5], [0.4104512516894041]]
        )
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[:3, 3:]
        assert_allclose(hess_logv_a, desired_hess_logv_a, atol=eps, rtol=rtol)

    def test_hess_logv_eta(self):
        desired_hess_logv_eta = np.array(
            [
                [-3.767308415119052e-16],
                [-8.901975977273483e-16],
                [1.266928439239253e-15],
            ]
        )
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[:3, 3:]
        assert_allclose(
            hess_logv_eta, desired_hess_logv_eta, atol=eps, rtol=rtol
        )

    def test_hess_delta_mu_delta_mu(self):
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )
        assert_allclose(hess_mu_mu, desired_hess_mu_mu, atol=eps, rtol=rtol)

    def test_hess_delta_mu_delta_a(self):
        n = self.n
        desired_hess_mu_a = [
            [-1.4104512516894041e00],
            [-1.7893015360387762e-16],
            [1.4104512516894041e00],
            [2.3462097484551508e-16],
        ]
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[:n, n:]
        assert_allclose(hess_mu_a, desired_hess_mu_a, atol=eps, rtol=rtol)

    def test_hess_delta_mu_eta(self):
        n = self.n
        desired_hess_mu_eta = [
            [1.0890338318440097e-16],
            [-3.7630114147090579e00],
            [-3.8494520984901552e-16],
            [3.7630114147090579e00],
        ]
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[:n, n:]
        assert_allclose(hess_mu_eta, desired_hess_mu_eta, atol=eps, rtol=rtol)

    def test_hess_delta_a_delta_a(self):
        desired_hess_a_a = [[0.3977033816628099]]
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )
        assert_allclose(hess_a_a, desired_hess_a_a, atol=eps, rtol=rtol)

    def test_hess_delta_a_eta(self):
        m = self.m
        desired_hess_a_eta = [[4.278233838022636e-16]]
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
            scale_eta_on_dt=self.scale_eta / self.dt,
        )[: m - 1, m - 1 :]
        assert_allclose(hess_a_eta, desired_hess_a_eta, atol=eps, rtol=rtol)

    def test_hess_eta_eta(self):
        desired_hess_eta_eta = [[47.28739606329804]]
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
            scale_eta_on_dt=self.scale_eta / self.dt,
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
        x,
        mu0,
        a0,
        eta0,
        fix_sigma_alpha,
        fix_sigma_beta,
        fix_sigma_tau,
        fix_mu,
        fix_a,
        fix_eta,
        pattern,
    ):
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
        sigma_alpha0,
        sigma_beta0,
        sigma_tau0,
        mu0,
        a0,
        eta0,
        scale_sigma_alpha,
        scale_sigma_beta,
        scale_sigma_tau,
        scale_delta_mu,
        scale_delta_a,
        scale_eta,
    ):
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
        fix_sigma_alpha,
        fix_sigma_beta,
        fix_sigma_tau,
        fix_mu,
        fix_a,
        fix_eta,
    ):
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

        m = self.m
        sigma = self.sigma
        sigma_est = np.asarray(
            [
                result.noise_model.sigma_alpha,
                result.noise_model.sigma_beta,
                result.noise_model.sigma_tau,
            ]
        ) * np.sqrt(m / (m - 1))
        assert_allclose(sigma_est / sigma, np.ones(3), atol=1e-1, rtol=1e-1)


class TestFit:
    n_even = 16
    dt_even = 1.0 / n_even
    t_even = np.arange(n_even) * dt_even
    f_even = np.fft.rfftfreq(n_even, dt_even)
    x_even = np.sin(4 * pi * t_even)
    p0_even = (0.5, dt_even)

    y_even_numpy_sign_true = transfer(
        tfun, x_even, dt=dt_even, args=p0_even, numpy_sign_convention=True
    )
    y_even_numpy_sign_false = transfer(
        tfun, x_even, dt=dt_even, args=p0_even, numpy_sign_convention=False
    )

    n_odd = 15
    dt_odd = 1.0 / n_odd
    t_odd = np.arange(n_odd) * dt_odd
    f_odd = np.fft.rfftfreq(n_odd, dt_odd)
    x_odd = np.sin(4 * pi * t_odd)
    p0_odd = (0.5, dt_odd)

    y_odd_numpy_sign_true = transfer(
        tfun, x_odd, dt=dt_odd, args=p0_odd, numpy_sign_convention=True
    )
    y_odd_numpy_sign_false = transfer(
        tfun, x_odd, dt=dt_odd, args=p0_odd, numpy_sign_convention=False
    )

    alpha, beta, tau = 1e-5, 0, 0
    sigma = np.array([alpha, beta, tau])

    @pytest.mark.parametrize("noise_parms", [(1, 0, 0), sigma**2])
    @pytest.mark.parametrize(
        "y, numpy_sign_convention",
        [(y_even_numpy_sign_true, True), (y_even_numpy_sign_false, False)],
    )
    @pytest.mark.parametrize(
        "f_bounds",
        [
            None,
            (-np.inf, np.inf),
            (-np.inf, f_even[-2]),
            (-np.inf, f_even[-3]),
            (f_even[0], np.inf),
            (f_even[0], f_even[-2]),
            (f_even[0], f_even[-3]),
            (f_even[1], np.inf),
            (f_even[1], f_even[-2]),
            (f_even[1], f_even[-3]),
        ],
    )
    @pytest.mark.parametrize("p_bounds", [None, ((0, -1), (2, 1))])
    @pytest.mark.parametrize("jac", [None, jac_fun])
    def test_inputs_even_length(
        self,
        y,
        noise_parms,
        numpy_sign_convention,
        f_bounds,
        p_bounds,
        jac,
    ):
        p0 = self.p0_even
        x = self.x_even
        dt = self.dt_even
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
        )
        assert_allclose(p.p_opt, p0)

    @pytest.mark.parametrize(
        "y_odd, numpy_sign_convention",
        [(y_odd_numpy_sign_true, True), (y_odd_numpy_sign_false, False)],
    )
    @pytest.mark.parametrize(
        "f_bounds",
        [
            None,
            (-np.inf, np.inf),
            (-np.inf, f_odd[-2]),
            (-np.inf, f_odd[-3]),
            (f_odd[0], np.inf),
            (f_odd[0], f_odd[-2]),
            (f_odd[0], f_odd[-3]),
            (f_odd[1], np.inf),
            (f_odd[1], f_odd[-2]),
            (f_odd[1], f_odd[-3]),
        ],
    )
    @pytest.mark.parametrize("jac", [None, jac_fun])
    def test_inputs_odd_length(self, y_odd, numpy_sign_convention, f_bounds, jac):
        p = fit(
            tfun,
            self.x_odd,
            y_odd,
            self.p0_odd,
            (1, 0, 0),
            dt=self.dt_odd,
            numpy_sign_convention=numpy_sign_convention,
            f_bounds=f_bounds,
            jac=jac,
        )
        assert_allclose(p.p_opt, self.p0_odd)

    def test_errors(self):
        p0 = self.p0_even
        x = self.x_even
        dt = self.dt_even
        y = self.y_even_numpy_sign_true

        with pytest.raises(
            ValueError, match="`bounds` must contain 2 elements."
        ):
            _ = fit(tfun, x, y, p0, dt=dt, p_bounds=())

        with pytest.raises(
            ValueError, match="sigma_parms must be a tuple of length"
        ):
            _ = fit(tfun, x, y, p0, (0, 0), dt=dt)
