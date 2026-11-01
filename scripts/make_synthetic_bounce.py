"""Render a synthetic bouncing-ball video with a known coefficient of restitution.

The ball is held still, dropped from rest, and bounces on a floor. Impacts happen
between frames (as in reality), the shutter blurs fast motion, and the sensor adds
noise. The true impact times are saved next to the video.

    python scripts/make_synthetic_bounce.py --out data/synthetic_bounce.mp4 --e 0.8 --drop 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galileo.bounce import simulate_drop  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic_bounce.mp4")
    p.add_argument("--e", type=float, default=0.8)
    p.add_argument("--e-slope", type=float, default=0.0, help="de/dT_prev, 1/s")
    p.add_argument("--drop", type=float, default=0.5, help="drop height, m")
    p.add_argument("--g", type=float, default=9.7975)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--release", type=float, default=0.7, help="time of release, s")
    p.add_argument("--px-per-m", type=float, default=1100.0)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    w, h = 1280, 720
    floor_y, radius, x0 = 680, 12, 520.0
    n = int(a.seconds * a.fps)
    t = np.arange(n) / a.fps
    sub = 6  # shutter open for half a frame, centred on the frame time, sampled 6 times
    t_fine = (t[:, None] + (np.arange(sub)[None, :] - (sub - 1) / 2) / (sub * 2 * a.fps)).ravel()
    t_fine = np.clip(t_fine, 0, None)
    height, impacts = simulate_drop(t_fine, a.drop, a.e, a.g, a.release, a.e_slope)
    height = height.reshape(n, sub)
    drift = 0.04 * np.clip(t_fine - a.release, 0, None).reshape(n, sub)  # slow sideways drift, m

    rng = np.random.default_rng(a.seed)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = (40 + 30 * xx / w + 15 * yy / h)[..., None] * np.ones(3)
    bg[floor_y + radius:] = 95  # floor
    bg += rng.normal(0, 3, (h, w, 1))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    truth = []
    for i in range(n):
        frame = np.zeros((h, w, 3))
        for k in range(sub):
            layer = bg.copy()
            cx = x0 + a.px_per_m * drift[i, k]
            cy = floor_y - a.px_per_m * height[i, k]
            cv2.circle(layer, (int(round(cx * 16)), int(round(cy * 16))), radius * 16,
                       (235, 235, 235), -1, cv2.LINE_AA, shift=4)
            frame += layer / sub
        frame += rng.normal(0, 2.5, frame.shape)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        truth.append([float(t[i]), float(x0 + a.px_per_m * drift[i].mean()),
                      float(floor_y - a.px_per_m * height[i].mean())])
    writer.release()

    meta = {"e": a.e, "e_slope_per_s": a.e_slope, "drop_height_m": a.drop, "g": a.g,
            "fps": a.fps, "release_s": a.release, "px_per_m": a.px_per_m,
            "impacts_s": [float(v) for v in impacts if v < t[-1]], "positions": truth}
    Path(a.out).with_suffix(".truth.json").write_text(json.dumps(meta))
    print(f"wrote {a.out}: {n} frames, true e = {a.e}, {len(meta['impacts_s'])} impacts in view")


if __name__ == "__main__":
    main()