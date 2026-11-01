"""The synthetic-video generator used for the accuracy map: is the truth file the truth?"""

import json

import cv2
import numpy as np

from galileo.synth import Scene, render


def test_rendered_bob_is_where_the_truth_says(tmp_path):
    s = Scene(width=320, height=240, px_per_m=400, length_m=0.4, bob_radius_px=8, seconds=1.0,
              noise=1.0, shutter=0.0, decoy=False, seed=3)
    truth = render(s, tmp_path / "v.mp4")
    assert json.loads((tmp_path / "v.truth.json").read_text())["frames"] == truth["frames"] == 30
    cap = cv2.VideoCapture(str(tmp_path / "v.mp4"))
    for i in (0, 10, 20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        assert ok
        x, y = truth["bob_xy"][i]
        patch = frame[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4].mean()
        assert patch > frame.mean() + 60  # bright bob exactly there


def test_rolling_shutter_and_blur_stretch_the_bob(tmp_path):
    sharp = Scene(width=320, height=240, px_per_m=400, bob_radius_px=8, seconds=0.5, noise=0,
                  shutter=0.0, decoy=False)
    blurred = Scene(**{**sharp.__dict__, "shutter": 1.0})
    a = render(sharp, tmp_path / "a.mp4")
    render(blurred, tmp_path / "b.mp4")

    def bright_area(path, i):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, f = cap.read()
        return int((f.mean(axis=2) > 150).sum())

    i = int(np.argmin([abs(x - 160) for x, _ in a["bob_xy"]]))  # fastest: at the bottom
    assert bright_area(tmp_path / "b.mp4", i) != bright_area(tmp_path / "a.mp4", i)