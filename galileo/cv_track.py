"""Fallback tracker without any model: median-background subtraction.

The pendulum bob is the only thing that moves, so the per-pixel median over
the clip is the empty background. |frame - background| -> threshold ->
morphological opening (erases the thin wire) -> biggest blob -> centroid.
Use it to sanity-check SAM 3.1, or when there is no network.
"""

from __future__ import annotations

import cv2
import numpy as np

from .track import Track


def _background(path: str, n_frames: int, max_samples: int = 120) -> np.ndarray:
    step = max(1, n_frames // max_samples)
    cap = cv2.VideoCapture(path)
    samples, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            samples.append(cv2.GaussianBlur(frame, (5, 5), 0))
        i += 1
    cap.release()
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def track_moving_blob(video_path: str, t: np.ndarray, diff_threshold: int = 30,
                      min_area: int = 15, open_px: int = 5) -> Track:
    n = len(t)
    bg = _background(str(video_path), n)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    x, y, area = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        diff = cv2.absdiff(cv2.GaussianBlur(frame, (5, 5), 0), bg).max(axis=2)
        _, fg = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        k, labels, stats, cents = cv2.connectedComponentsWithStats(fg)
        if k <= 1:
            continue
        cand = [j for j in range(1, k) if stats[j, cv2.CC_STAT_AREA] >= min_area]
        if not cand:
            continue
        if prev is None:
            j = max(cand, key=lambda j: stats[j, cv2.CC_STAT_AREA])
        else:  # prefer big blobs near the previous position
            j = max(cand, key=lambda j: stats[j, cv2.CC_STAT_AREA]
                    / (1.0 + np.hypot(*(cents[j] - prev)) / 50.0))
        m = labels == j
        ys, xs = np.nonzero(m)
        x[i], y[i], area[i] = xs.mean(), ys.mean(), float(len(xs))
        prev = np.array([x[i], y[i]])
    cap.release()
    return Track(t, x, y, area, np.full(n, np.nan), source="opencv-bgsub")
