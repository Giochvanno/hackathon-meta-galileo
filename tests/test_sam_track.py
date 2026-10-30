import numpy as np

from galileo.pendulum import analyze_pendulum
from galileo.sam_track import choose_moving_object, parse_sam_output, tracks_from_detections

from .helpers import DOC_LINE, fake_sam_text, simulate_pendulum


def test_parses_documented_example():
    dets, warns = parse_sam_output(DOC_LINE)
    assert not warns
    assert [d.object_id for d in dets] == ["0", "1"]
    for d in dets:
        left, top, right, bottom = d.box
        assert left <= d.cx < right and top <= d.cy < bottom
        assert d.frame == 0 and d.area > 0 and np.isnan(d.conf)


def test_skipped_frames_become_nan_and_moving_object_is_chosen():
    t = np.arange(40) / 30
    xs = 600 + 50 * np.sin(2 * np.pi * t)
    text = fake_sam_text(xs, np.full_like(xs, 500.0), drop_every=10)
    dets, _ = parse_sam_output(text)
    tracks = tracks_from_detections(dets, t)
    best, summary = choose_moving_object(tracks)
    assert best == "3" and summary["7"]["spread_px"] < 1
    assert np.isnan(tracks["3"].x[5]) and np.isfinite(tracks["3"].x[6])
    assert np.nanmax(np.abs(tracks["3"].x - xs)) <= 0.51  # integer boxes: <= half a pixel
    assert np.nanmin(tracks["3"].conf) > 0.9


def test_malformed_line_is_reported_not_fatal():
    dets, warns = parse_sam_output(DOC_LINE + "<1f>0<|box;x1=1|>junk\n")
    assert len(dets) == 2 and warns


def test_end_to_end_fake_sam_to_g():
    g, L, ppm = 9.7975, 0.52, 1100.0
    t = np.arange(600) / 30
    th = simulate_pendulum(t, g, L)
    xs, ys = 640 + ppm * L * np.sin(th), 30 + ppm * L * np.cos(th)
    dets, _ = parse_sam_output(fake_sam_text(xs, ys, drop_every=25))
    best, _ = choose_moving_object(tracks := tracks_from_detections(dets, t))
    res = analyze_pendulum(tracks[best].clean(), L, 0.002, g_reference=g)
    assert abs(res.g - g) / g < 0.003
    assert res.px_per_m is not None and abs(res.px_per_m - ppm) / ppm < 0.1


def test_only_static_objects_raises():
    import pytest

    from galileo.sam_track import NoMovingObjectError

    t = np.arange(40) / 30
    text = fake_sam_text(np.full(40, 600.0), np.full(40, 500.0))
    tracks = tracks_from_detections(parse_sam_output(text)[0], t)
    with pytest.raises(NoMovingObjectError):
        choose_moving_object(tracks)
