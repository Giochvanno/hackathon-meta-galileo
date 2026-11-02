"""Any oscillator: period, frequency and damping without a length."""

import numpy as np
import pytest

from galileo.oscillator import analyze_oscillation, oscillation_report
from galileo.track import Track


def spring_track(f=2.3, gamma=0.15, fps=30.0, seconds=12.0, noise=0.4, seed=0, vertical=True):
    rng = np.random.default_rng(seed)
    n = int(fps * seconds)
    t = np.sort(np.arange(n) / fps + rng.normal(0, 0.0007, n))
    w = 2 * np.pi * np.sqrt(f**2 + (gamma / (2 * np.pi)) ** 2)  # undamped omega for this f
    wd = 2 * np.pi * f
    s = 60 * np.exp(-gamma * t) * np.cos(wd * t + 0.4)
    x = 300 + (0 if vertical else s) + rng.normal(0, noise, n)
    y = 400 + (s if vertical else 0) + rng.normal(0, noise, n)
    nan = np.full(n, np.nan)
    return Track(t, x, y, nan, nan, "sim"), w


def test_a_spring_bobbing_up_and_down(tmp_path):
    tr, _ = spring_track()
    r = analyze_oscillation(tr)
    assert r.reliable and r.best_model == "viscous"
    assert r.frequency_hz == pytest.approx(2.3, abs=4 * r.sigma_frequency_hz + 1e-4)
    assert r.damping_per_s == pytest.approx(0.15, rel=0.1)
    assert r.quality_factor == pytest.approx(2 * np.pi * 2.3 / (2 * 0.15), rel=0.1)
    assert abs(r.axis[1]) > 0.99  # the motion is vertical in the image
    oscillation_report(tr, r, tmp_path / "osc.png")
    assert (tmp_path / "osc.png").exists()


def test_fast_vibration_close_to_the_frame_rate_limit():
    tr, _ = spring_track(f=11.0, gamma=0.3, fps=60.0, seconds=6.0, vertical=False)
    r = analyze_oscillation(tr)
    assert r.frequency_hz == pytest.approx(11.0, rel=1e-3)


def test_something_that_does_not_oscillate_is_refused():
    t = np.arange(300) / 30
    rng = np.random.default_rng(1)
    nan = np.full(300, np.nan)
    tr = Track(t, 300 + rng.normal(0, 0.6, 300), 200 + rng.normal(0, 0.6, 300), nan, nan, "sim")
    r = analyze_oscillation(tr)
    assert not r.reliable
