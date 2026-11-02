"""The no-click pipeline end to end on a synthetic video (Spark and SAM replaced offline)."""

import json

import pytest

from galileo.assistant import Look
from galileo.synth import Scene, render


def test_pendulum_measurement_without_clicks(tmp_path, monkeypatch):
    import galileo.assistant as assistant
    import galileo.cli as cli
    from galileo import pipeline

    s = Scene(width=640, height=360, px_per_m=500, length_m=0.40, bob_radius_px=10,
              seconds=12.0, seed=2)
    truth = render(s, tmp_path / "v.mp4")
    x, y = (int(v) for v in truth["bob_xy"][0])

    def fake_look(video_path, t, out, refresh=False):  # what Muse Spark would answer
        return Look("pendulum", "white bob", 0, (x, y), 0, 0.0, None, None, [], "high", "ok")

    real_track = cli._track_video

    def offline_track(a):  # the model-free tracker instead of SAM 3.1
        a.tracker = "opencv"
        return real_track(a)

    monkeypatch.setattr(assistant, "look_cached", fake_look)
    monkeypatch.setattr(cli, "_track_video", offline_track)
    steps, charged = [], []
    summary = pipeline.measure(tmp_path / "v.mp4", tmp_path / "out", length_m=0.40,
                               step=lambda n, d="": steps.append(n), budget_check=charged.append)
    assert summary["experiment"] == "pendulum"
    assert summary["result"]["g"] == pytest.approx(s.g, rel=1e-3) and summary["result"]["reliable"]
    assert steps[:4] == ["prepare", "look", "track", "physics"] and steps[-1] == "done"
    assert charged and sum(charged) < 0.2  # the budget was asked before paying
    assert "report.png" in summary["files"]
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["experiment"] == "pendulum"


def test_a_pendulum_without_a_length_still_gives_period_and_damping(tmp_path, monkeypatch):
    import galileo.assistant as assistant
    import galileo.cli as cli
    from galileo import pipeline

    s = Scene(width=640, height=360, px_per_m=500, length_m=0.40, bob_radius_px=10,
              seconds=12.0, seed=3)
    truth = render(s, tmp_path / "v.mp4")
    x, y = (int(v) for v in truth["bob_xy"][0])
    monkeypatch.setattr(assistant, "look_cached", lambda *a, **k: Look(
        "pendulum", "bob", 0, (x, y), 0, 0.0, None, None, [], "high", ""))
    real_track = cli._track_video

    def offline_track(a):
        a.tracker = "opencv"
        return real_track(a)

    monkeypatch.setattr(cli, "_track_video", offline_track)
    summary = pipeline.measure(tmp_path / "v.mp4", tmp_path / "out")
    assert summary["experiment"] == "oscillator" and summary["notes"]
    assert summary["result"]["period_s"] == pytest.approx(truth["T0_small_angle"], rel=0.01)
    assert "oscillation_report.png" in summary["files"]
