"""Synthetic pendulum videos with a known g, and controlled ways of making them worse.

Used to measure how accurate the instrument is, and where it breaks: motion blur,
sensor noise, a small or low-contrast bob, low frame rate, short recordings, a
shaking or tilted phone, rolling shutter, large swings, strong damping.

The motion is the exact nonlinear damped pendulum. Each pixel row is exposed over
the shutter time and (with a rolling shutter) later for lower rows, so blur and
skew come from the physics, not from an image filter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Scene:
    g: float = 9.7975
    length_m: float = 0.40
    theta0_deg: float = 12.0
    gamma: float = 0.02            # viscous damping rate, 1/s
    seconds: float = 15.0
    fps: float = 30.0
    width: int = 1280
    height: int = 720
    px_per_m: float = 1100.0
    bob_radius_px: float = 20.0
    shutter: float = 0.5           # fraction of the frame interval the sensor is exposed
    noise: float = 3.0             # sensor noise, grey levels (1 sigma)
    contrast: float = 120.0        # bob brightness above the background, grey levels
    shake_px: float = 0.0          # camera shake (random walk), px
    rolling_shutter_ms: float = 0.0  # read-out time from the top row to the bottom row
    tilt_deg: float = 0.0          # phone rolled: the whole image is rotated
    decoy: bool = True             # a second, motionless bob in the frame
    seed: int = 0

    @property
    def pivot(self) -> np.ndarray:
        return np.array([self.width / 2, 0.12 * self.height])


def _motion(s: Scene):
    def rhs(_, z):
        return [z[1], -s.g / s.length_m * np.sin(z[0]) - 2 * s.gamma * z[1]]

    sol = solve_ivp(rhs, (-1.0, s.seconds + 1.0), [np.radians(s.theta0_deg), 0.0],
                    dense_output=True, rtol=1e-10, atol=1e-12, t_eval=None,
                    first_step=1e-3, max_step=0.01)
    return lambda t: sol.sol(t)[0]


def _background(s: Scene, rng) -> np.ndarray:
    h, w = s.height, s.width
    yy, xx = np.mgrid[0:h, 0:w]
    bg = (70 + 30 * xx / w + 20 * yy / h)[..., None] * np.array([1.0, 1.02, 1.05], np.float32)
    # Static texture: features for the camera-motion estimate, and clutter for the trackers.
    for _ in range(40):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        col = rng.uniform(40, 140, 3)
        if rng.random() < 0.5:
            cv2.rectangle(bg, (x0, y0), (x0 + int(rng.integers(8, 60)), y0 + int(rng.integers(8, 60))),
                          col.tolist(), -1)
        else:
            cv2.circle(bg, (x0, y0), int(rng.integers(3, 15)), col.tolist(), -1)
    bg[int(0.12 * h) - 6:int(0.12 * h)] = 55  # the shelf the pendulum hangs from
    bg += rng.normal(0, 4, (h, w, 1))
    return bg.astype(np.float32)


def _bob_alpha(s: Scene, theta, t_frame: float, box) -> np.ndarray:
    """Fraction of exposure each pixel in `box` sees the bob, with shutter and rolling shutter."""
    x0, y0, x1, y1 = box
    rows = np.arange(y0, y1)[None, :, None] + 0.5
    cols = np.arange(x0, x1)[None, None, :] + 0.5
    sub = 6
    k = (np.arange(sub) - (sub - 1) / 2) / sub * s.shutter / s.fps
    t = t_frame + k[:, None, None] + s.rolling_shutter_ms / 1e3 * rows / s.height
    th = theta(t.ravel()).reshape(t.shape)
    r = s.length_m * s.px_per_m
    cx = s.pivot[0] + r * np.sin(th)
    cy = s.pivot[1] + r * np.cos(th)
    d = np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2)
    return np.clip(s.bob_radius_px + 0.5 - d, 0, 1).mean(axis=0)


def render(s: Scene, out: str | Path) -> dict:
    """Write the video and a truth file next to it. Returns the truth."""
    rng = np.random.default_rng(s.seed)
    theta = _motion(s)
    bg = _background(s, rng)
    if s.decoy:
        cv2.circle(bg, (int(0.82 * s.width), int(0.85 * s.height)), int(s.bob_radius_px),
                   (np.array([70, 72, 75]) + s.contrast).tolist(), -1, cv2.LINE_AA)
    n = int(round(s.seconds * s.fps))
    t = np.arange(n) / s.fps
    r = s.length_m * s.px_per_m
    pad = s.bob_radius_px + 4 + r * np.radians(abs(s.theta0_deg)) * 2 * np.pi * np.sqrt(
        s.g / s.length_m) * s.shutter / s.fps
    # Camera shake: a smooth random walk, then a roll of the whole picture.
    walk = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
    walk -= walk.mean(axis=0)
    if s.shake_px > 0 and np.abs(walk).max() > 0:
        walk *= s.shake_px / np.abs(walk).max()
    else:
        walk[:] = 0
    centre = (s.width / 2, s.height / 2)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")  # renamed when complete: no half-written videos
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), s.fps, (s.width, s.height))
    truth_xy = []
    bob_col = (np.array([70, 72, 75]) + s.contrast).astype(np.float32)
    # Sensor noise: a bank of independent noise images, picked at random each frame (fast).
    bank = rng.normal(0, 1, (12, s.height, s.width, 3)).astype(np.float32) * s.noise
    for i in range(n):
        th = float(theta(t[i]))
        bx, by = s.pivot + r * np.array([np.sin(th), np.cos(th)])
        frame = bg.copy()
        cv2.line(frame, tuple(int(v) for v in s.pivot), (int(bx), int(by)), (60, 60, 60), 1,
                 cv2.LINE_AA)
        x0, x1 = int(max(bx - pad, 0)), int(min(bx + pad + 1, s.width))
        y0, y1 = int(max(by - pad, 0)), int(min(by + pad + 1, s.height))
        a = _bob_alpha(s, theta, t[i], (x0, y0, x1, y1))[..., None]
        frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - a) + bob_col * a
        m = cv2.getRotationMatrix2D(centre, s.tilt_deg, 1.0)
        m[:, 2] += walk[i]
        if s.tilt_deg or s.shake_px:
            frame = cv2.warpAffine(frame, m, (s.width, s.height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
        noise = bank[rng.integers(len(bank))]
        frame += np.roll(noise, (int(rng.integers(s.height)), int(rng.integers(s.width))), (0, 1))
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        truth_xy.append((m @ np.array([bx, by, 1.0])).tolist())
    writer.release()
    tmp.replace(out)

    truth = {**asdict(s), "T0_small_angle": float(2 * np.pi * np.sqrt(s.length_m / s.g)),
             "bob_xy": truth_xy, "frames": n}
    out.with_suffix(".truth.json").write_text(json.dumps(truth))
    return truth