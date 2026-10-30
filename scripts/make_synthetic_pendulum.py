"""Render a synthetic pendulum video with a known g, to test the pipeline end to end.

The motion is the exact nonlinear pendulum with viscous damping (solved
numerically), so the analysis has to get the finite-amplitude correction right.
A second, motionless "earbud" sits in the frame to test object selection.

    python scripts/make_synthetic_pendulum.py --out data/synthetic.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp


def simulate(t: np.ndarray, g: float, length: float, theta0_deg: float, gamma: float) -> np.ndarray:
    def rhs(_, z):
        return [z[1], -g / length * np.sin(z[0]) - 2 * gamma * z[1]]

    sol = solve_ivp(rhs, (0, t[-1] + 1), [np.radians(theta0_deg), 0.0], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    return sol.y[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic.mp4")
    p.add_argument("--g", type=float, default=9.7975)
    p.add_argument("--length", type=float, default=0.52)
    p.add_argument("--theta0", type=float, default=13.0)
    p.add_argument("--gamma", type=float, default=0.025)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--px-per-m", type=float, default=1100.0)
    a = p.parse_args()

    w, h = 1280, 720
    pivot = np.array([640.0, 30.0])
    n = int(a.seconds * a.fps)
    t = np.arange(n) / a.fps
    sub = 4  # motion blur: average 4 positions over half a frame (1/60 s shutter at 30 fps)
    t_fine = (t[:, None] + np.arange(sub)[None, :] / (sub * 2 * a.fps)).ravel()
    theta = simulate(t_fine, a.g, a.length, a.theta0, a.gamma).reshape(n, sub)
    r_px = a.length * a.px_per_m

    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    bg = (30 + 25 * xx / w + 10 * yy / h)[..., None] * np.ones(3)
    bg += rng.normal(0, 4, (h, w, 1))
    cv2.circle(bg, (1080, 650), 11, (225, 225, 225), -1, cv2.LINE_AA)  # the other earbud, at rest

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    truth = []
    for i in range(n):
        frame = np.zeros((h, w, 3))
        for k in range(sub):
            layer = bg.copy()
            th = theta[i, k]
            bob = pivot + r_px * np.array([np.sin(th), np.cos(th)])
            cv2.line(layer, tuple(np.round(pivot).astype(int)), tuple(np.round(bob).astype(int)),
                     (90, 90, 90), 1, cv2.LINE_AA)
            cv2.ellipse(layer, (int(round(bob[0] * 16)), int(round(bob[1] * 16))), (9 * 16, 13 * 16),
                        np.degrees(-th), 0, 360, (240, 240, 240), -1, cv2.LINE_AA, shift=4)
            frame += layer / sub
        frame += rng.normal(0, 2.5, frame.shape)  # sensor noise
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        th = theta[i].mean()
        truth.append([float(t[i]), *(pivot + r_px * np.array([np.sin(th), np.cos(th)])).tolist()])
    writer.release()

    T0 = 2 * np.pi * np.sqrt(a.length / a.g)
    meta = {"g": a.g, "length_m": a.length, "theta0_deg": a.theta0, "gamma": a.gamma,
            "fps": a.fps, "px_per_m": a.px_per_m, "pivot_px": pivot.tolist(),
            "T0_small_angle": T0, "size": [w, h], "positions": truth}
    Path(a.out).with_suffix(".truth.json").write_text(json.dumps(meta))
    print(f"wrote {a.out}: {n} frames, true g = {a.g}, T0 = {T0:.4f} s")


if __name__ == "__main__":
    main()
