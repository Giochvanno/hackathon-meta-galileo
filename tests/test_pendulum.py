import numpy as np
import pytest

from galileo.pendulum import (analyze_pendulum, fit_models, local_g, predict_period,
                              zero_crossing_period)
from galileo.track import Track

from .helpers import simulate_pendulum


def _track(t, theta, length=0.52, ppm=1000.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = 640 + ppm * length * np.sin(theta) + rng.normal(0, noise, len(t))
    y = 40 + ppm * length * np.cos(theta) + rng.normal(0, noise, len(t))
    nan = np.full(len(t), np.nan)
    return Track(t, x, y, nan.copy(), nan.copy())


@pytest.mark.parametrize("theta0", [6.0, 13.0, 20.0])
def test_recovers_g_from_nonlinear_damped_pendulum(theta0):
    g, L = 9.7975, 0.52
    t = np.arange(0, 20, 1 / 30)
    res = analyze_pendulum(_track(t, simulate_pendulum(t, g, L, theta0)), L, 0.002, g_reference=g)
    assert res.best_model == "viscous"
    assert abs(res.g - g) / g < 0.002          # within 0.2 %
    assert abs(res.g - g) < 3 * res.sigma_g
    assert res.amplitude_source == "arc fit"
    assert abs(res.amplitude_deg - theta0) < 1.5


def test_variable_frame_rate_timestamps():
    """Phone VFR: uneven sampling must not bias the period."""
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(1 / 36, 1 / 26, 600))
    t -= t[0]
    res = analyze_pendulum(_track(t, simulate_pendulum(t)), 0.52, 0.002, g_reference=9.7975)
    assert abs(res.g - 9.7975) / 9.7975 < 0.002


def test_zero_crossings_agree_with_fit():
    t = np.arange(0, 20, 1 / 30)
    res = analyze_pendulum(_track(t, simulate_pendulum(t)), 0.52)
    assert res.crossings["n_crossings"] >= 20
    assert res.consistency_z < 3


def test_model_selection_distinguishes_drag_laws():
    t = np.arange(0, 30, 1 / 30)
    tau = t - t.mean()
    w = 2 * np.pi / 1.45
    rng = np.random.default_rng(1)
    hyperbolic = 120 / (1 + 0.04 * (t - t[0])) * np.cos(w * t) + rng.normal(0, 0.4, len(t))
    exponential = 120 * np.exp(-0.04 * t) * np.cos(w * t) + rng.normal(0, 0.4, len(t))
    assert fit_models(tau, hyperbolic, w)[0].name == "quadratic"
    assert fit_models(tau, exponential, w)[0].name == "viscous"


def test_zero_crossing_period_on_clean_sine():
    t = np.linspace(0, 10, 1000)
    out = zero_crossing_period(t, np.sin(2 * np.pi * t / 1.25), 0.3)
    assert out["T"] == pytest.approx(1.25, abs=1e-4)


def test_predict_period():
    T, s = predict_period(0.30, 9.80, 0.05, 0.003)
    assert T == pytest.approx(2 * np.pi * np.sqrt(0.30 / 9.80), rel=1e-9)
    assert 0 < s < 0.01


def test_local_g():
    assert local_g(0) == pytest.approx(9.7803, abs=1e-3)
    assert local_g(90) == pytest.approx(9.8322, abs=1e-3)
    assert local_g(35.23) == pytest.approx(9.7975, abs=1e-3)


def test_too_few_points_raises():
    t = np.arange(0, 1, 1 / 30)
    with pytest.raises(ValueError):
        analyze_pendulum(_track(t, simulate_pendulum(t)), 0.52)


def test_static_object_is_flagged_unreliable():
    rng = np.random.default_rng(5)
    t = np.arange(0, 25, 1 / 30)
    n = len(t)
    tr = Track(t, 335 + rng.normal(0, 1, n), 150 + rng.normal(0, 1, n), np.full(n, np.nan),
               np.full(n, np.nan))
    res = analyze_pendulum(tr, 0.46)
    assert not res.reliable and res.problems
