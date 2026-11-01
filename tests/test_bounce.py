"""Bouncing ball: impact timing, restitution models, predictions, and guards (simulated drops)."""

import numpy as np
import pytest

from galileo.bounce import (analyze_bounce, fit_restitution, predict_first_flight,
                            predict_impacts, simulate_drop)
from galileo.track import Track


def drop_track(fps=30.0, h0=0.5, e=0.8, slope=0.0, noise=0.4, seed=1, seconds=4.0,
               px_per_m=600.0, t_release=0.7):
    """A phone-like recording: jittered frame times, ball held still, dropped, bouncing."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    t = np.sort(np.arange(n) / fps + 0.013 + rng.normal(0, 0.0007, n))
    h, impacts = simulate_drop(t, h0, e, t_release=t_release, e_slope_per_s=slope)
    y = 650 - px_per_m * h + rng.normal(0, noise, n)
    x = 400 + 20 * t + rng.normal(0, noise, n)  # a little sideways drift, as in real drops
    nan = np.full(n, np.nan)
    return Track(t, x, y, nan, nan, "sim"), impacts


def test_impacts_are_timed_between_frames_and_e_is_recovered():
    tr, truth = drop_track(fps=30)
    r = analyze_bounce(tr, drop_height_m=0.5)
    assert r.reliable and len(r.impacts_s) >= 4
    err = np.array(r.impacts_s) - truth[: len(r.impacts_s)]
    assert np.max(np.abs(err)) < 0.003           # frames are 33 ms apart; impacts known to ~1 ms
    assert np.all(np.abs(err) < 4 * np.array(r.sigma_impacts_s))
    assert r.best_model == "constant"
    assert abs(r.e - 0.8) < 3 * r.sigma_e and r.sigma_e < 0.002
    # Drop height gives a pixel scale, and the flights' curvature then gives g.
    assert r.g_from_scale == pytest.approx(9.7975, rel=0.01)


def test_error_bars_are_honest():
    pulls = []
    for seed in range(60):
        tr, _ = drop_track(seed=100 + seed)
        r = analyze_bounce(tr)
        pulls.append((r.e - 0.8) / r.sigma_e)
    assert 0.7 < np.std(pulls) < 1.4 and abs(np.mean(pulls)) < 0.4


def test_speed_dependent_restitution_is_detected():
    tr, truth = drop_track(fps=60, slope=-0.3, seed=2)
    r = analyze_bounce(tr, drop_height_m=0.5)
    assert r.best_model == "speed-dependent"
    assert r.e_slope_per_s == pytest.approx(-0.3, abs=4 * r.sigma_e_slope + 0.01)
    const = next(m for m in r.models if m["name"] == "constant")
    assert const["delta_aic"] > 10
    # The fitted speed law predicts the first flight of the drop it never saw directly.
    assert r.predicted_first_flight_s == pytest.approx(truth[1] - truth[0],
                                                       abs=3 * r.sigma_predicted_first_flight_s)


def test_prediction_for_a_new_drop_height():
    tr, _ = drop_track(h0=0.5, seed=3)
    r = analyze_bounce(tr)
    fit = fit_restitution(np.array(r.impacts_s), np.array(r.sigma_impacts_s))[0]
    T, s = predict_first_flight(0.30, fit, sigma_drop_m=0.003)
    _, truth = simulate_drop(np.linspace(0, 3, 10), 0.30, 0.8)
    assert abs(T - (truth[1] - truth[0])) < 2 * s and s < 0.005


def test_future_impacts_from_the_first_few():
    tr, truth = drop_track(fps=60, seed=4)
    r = analyze_bounce(tr)
    seen = np.array(r.impacts_s[:4])
    fit = fit_restitution(seen, np.array(r.sigma_impacts_s[:4]))[0]
    future = predict_impacts(seen, fit, 3)
    assert np.allclose(future, truth[4:7], atol=0.01)


def test_still_object_is_refused():
    t = np.arange(120) / 30
    rng = np.random.default_rng(0)
    nan = np.full(120, np.nan)
    tr = Track(t, 300 + rng.normal(0, 0.5, 120), 400 + rng.normal(0, 0.5, 120), nan, nan, "sim")
    with pytest.raises(ValueError, match="impact"):
        analyze_bounce(tr)


def test_wrong_object_part_of_the_time_is_not_trusted():
    tr, _ = drop_track(seed=5)
    y = tr.y.copy()
    y[60:] = 300 + 40 * np.sin(np.arange(len(y) - 60) * 0.9)  # tracker jumped to something else
    bad = Track(tr.t, tr.x, y, tr.area, tr.conf, "sim")
    try:
        r = analyze_bounce(bad)
    except ValueError:
        return
    assert not r.reliable or len(r.impacts_s) < len(analyze_bounce(tr).impacts_s)


def test_cli_on_a_track_csv(tmp_path, capsys):
    from galileo.cli import main

    tr, _ = drop_track(seed=6)
    csv = tmp_path / "track.csv"
    tr.to_csv(csv)
    main(["bounce", str(csv), "--drop-height", "0.5", "--predict-height", "0.3"])
    out = capsys.readouterr().out
    e = float(out.split("coefficient of restitution e = ")[1].split()[0])
    assert abs(e - 0.8) < 0.003 and "drop from 30 cm" in out
    assert (tmp_path / "bounce_report.png").exists() and (tmp_path / "result_bounce.json").exists()


def test_first_flight_when_the_drop_is_not_seen():
    """Release out of frame: the first flight starts where its parabola leaves the floor level."""
    tr, truth = drop_track(seed=7)
    y = tr.y.copy()
    y[tr.t < truth[0] - 0.08] = np.nan  # only the last 2-3 frames of the fall are visible
    r = analyze_bounce(Track(tr.t, tr.x, y, tr.area, tr.conf, "sim"), drop_height_m=0.5)
    assert r.first_flight_method == "floor level"
    assert abs(r.first_flight_s - (truth[1] - truth[0])) < max(3 * r.sigma_first_flight_s, 0.004)
    assert abs(r.first_flight_s - r.predicted_first_flight_s) < 0.01


def test_recording_that_starts_mid_bounce_is_not_taken_for_the_drop():
    tr, truth = drop_track(seed=8)
    y = tr.y.copy()
    y[tr.t < truth[0] + 0.03] = np.nan  # starts just after the first impact
    r = analyze_bounce(Track(tr.t, tr.x, y, tr.area, tr.conf, "sim"), drop_height_m=0.5)
    assert r.first_flight_s is None and not r.first_impact_is_drop and r.g_from_scale is None
    assert abs(r.e - 0.8) < 3 * r.sigma_e


def test_rolling_shutter_is_measured_and_corrected():
    """Rows lower in the image are read later (25 ms over 720 rows, like a phone sensor)."""
    rng = np.random.default_rng(9)
    fps, k, ppm = 30.0, 0.025 / 720, 600.0
    n = int(4 * fps)
    t = np.arange(n) / fps + 0.013
    y = np.full(n, 300.0)
    for _ in range(30):  # the row a point lands on depends on when it is read: iterate
        h, truth = simulate_drop(t + k * y, 0.5, 0.8, t_release=0.7)
        y = 650 - ppm * h
    y = y + rng.normal(0, 0.4, n)
    nan = np.full(n, np.nan)
    tr = Track(t, 400 + rng.normal(0, 0.4, n), y, nan, nan, "sim")
    r = analyze_bounce(tr, frame_height_px=720)
    assert r.reliable
    assert r.rolling_shutter_readout_ms == pytest.approx(25, abs=4)
    assert abs(r.e - 0.8) < 3 * r.sigma_e
    off = np.array(r.flights_s) - np.diff(truth)[: len(r.flights_s)]
    assert np.max(np.abs(off)) < 0.002