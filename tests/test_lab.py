import numpy as np
import pytest

from galileo.lab import Lab, fit_length_series
from galileo.track import Track

from .helpers import simulate_pendulum

G = 9.7975


def write_track(path, length_eff, seconds=20.0, theta0=10.0, ppm=1000.0, seed=0, still=False):
    """A track.csv of a pendulum whose *true* length is length_eff."""
    t = np.arange(0, seconds, 1 / 30)
    rng = np.random.default_rng(seed)
    th = np.zeros_like(t) if still else simulate_pendulum(t, G, length_eff, theta0)
    x = 640 + ppm * length_eff * np.sin(th) + rng.normal(0, 0.5, len(t))
    y = 40 + ppm * length_eff * np.cos(th) + rng.normal(0, 0.5, len(t))
    nan = np.full(len(t), np.nan)
    Track(t, x, y, nan.copy(), nan.copy()).to_csv(path)
    return path


def test_series_fit_recovers_g_and_offset():
    L = np.array([0.25, 0.35, 0.45, 0.55])
    offset = 0.012  # tape stops 12 mm short of the centre of mass
    T0 = 2 * np.pi * np.sqrt((L + offset) / G)
    fit = fit_length_series(L, np.full(4, 0.001), T0, np.full(4, 0.0005), fit_offset=True)
    assert fit.g == pytest.approx(G, rel=2e-3)
    assert fit.offset_m == pytest.approx(offset, abs=1e-3)
    T, s = fit.predict_T0(0.30, 0.001)
    assert T == pytest.approx(2 * np.pi * np.sqrt(0.312 / G), rel=1e-3) and 0 < s < 0.01


def test_series_fit_without_offset_is_weighted_mean():
    L = np.array([0.39, 0.39])
    T0 = 2 * np.pi * np.sqrt(L / G)
    fit = fit_length_series(L, [0.003, 0.003], T0, [0.001, 0.001], fit_offset=False)
    assert fit.g == pytest.approx(G, rel=1e-6) and fit.offset_m == 0.0


def test_prediction_is_frozen_and_verified_blind(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    e1 = lab.add_experiment(write_track(tmp_path / "a.csv", 0.40, seed=1), 0.40, 0.002)
    lab.estimate_g([e1], fit_offset=False)
    p = lab.predict(0.30, length_err_m=0.002)
    # A real experiment at 0.30 m arrives after the prediction.
    e2 = lab.add_experiment(write_track(tmp_path / "b.csv", 0.30, seed=2), 0.30, 0.002)
    lab.estimate_g([e1, e2], fit_offset=False)  # model changes; the prediction must not
    v = lab.verify(p["id"], e2)
    assert v["blind"] and v["verdict"] == "confirmed" and abs(v["z"]) < 2
    assert lab.state["predictions"]["P1"]["model"]["basis"] == [e1]

    # Testing a prediction against an experiment it was built from is not blind.
    assert lab.verify(p["id"], e1)["blind"] is False


def test_wrong_prediction_is_refuted(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    e1 = lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.001)
    lab.estimate_g([e1], fit_offset=False)
    p = lab.predict(0.30, length_err_m=0.001)
    e2 = lab.add_experiment(write_track(tmp_path / "b.csv", 0.36, seed=3), 0.30, 0.001)
    assert lab.verify(p["id"], e2)["verdict"] == "refuted"


def test_unreliable_experiments_are_left_out(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    good = lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002)
    bad = lab.add_experiment(write_track(tmp_path / "b.csv", 0.40, still=True), 0.40, 0.002)
    out = lab.estimate_g([good, bad], fit_offset=False)
    assert out["basis"] == [good] and out["rejected_unreliable"] == [bad]


def test_offset_needs_three_lengths(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    e1 = lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002)
    with pytest.raises(ValueError):
        lab.estimate_g([e1], fit_offset=True)


def test_notebook_persists(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002, note="first")
    again = Lab(tmp_path / "lab.json")
    assert again.experiment("E1")["note"] == "first"