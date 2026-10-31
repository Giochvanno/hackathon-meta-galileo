import json

import cv2
import numpy as np

from galileo.render import _running_period, render_overlay

from .helpers import fake_sam_text


def test_running_period():
    assert _running_period([0.1, 0.6], 1.0) == (None, 2)
    T, n = _running_period([0.1, 0.6, 1.1, 1.6], 5.0)
    assert n == 4 and abs(T - 1.0) < 1e-12


def test_render_overlay_writes_every_frame(tmp_path):
    n, w, h = 40, 320, 240
    t = np.arange(n) / 30
    xs = 160 + 60 * np.sin(2 * np.pi * t)
    ys = np.full(n, 150.0)
    vw = cv2.VideoWriter(str(tmp_path / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for x in xs:
        f = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(f, (int(x), 150), 12, (240, 240, 240), -1)
        vw.write(f)
    vw.release()
    (tmp_path / "frame_times.json").write_text(json.dumps(t.tolist()))
    (tmp_path / "sam_point.txt").write_text(fake_sam_text(xs, ys, size=(w, h)))
    result = {"g": 9.8, "sigma_g": 0.05, "T0": 1.0, "amplitude_source": "user", "pivot_px": None,
              "crossings": {"crossing_times": [0.25, 0.75, 1.25]}}
    (tmp_path / "result.json").write_text(json.dumps(
        {"result": result, "video": {}, "track_meta": {"object_id": "3"}}))

    out = render_overlay(tmp_path)
    cap = cv2.VideoCapture(str(out))
    frames = 0
    while cap.read()[0]:
        frames += 1
    assert frames == n