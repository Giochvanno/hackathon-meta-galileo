"""Camera-motion compensation, so a hand-held or leaning phone still measures correctly.

For each frame we estimate a similarity transform (shift, rotation, zoom) of the
static scene relative to frame 0: corners are tracked frame to frame with
pyramidal Lucas-Kanade, and RANSAC throws out points on the moving pendulum.
Tracked positions are then mapped back into frame-0 coordinates.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .track import Track


def estimate_camera_motion(video_path: str | Path, max_corners: int = 400) -> np.ndarray:
    """Array (n, 2, 3): affine map from frame-i pixels to frame-0 pixels."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video_path}")
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_cur = np.eye(3)                    # frame 0 -> current frame
    maps = [np.eye(3)[:2].copy()]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        step = np.eye(3)
        p0 = cv2.goodFeaturesToTrack(prev, max_corners, 0.01, 12)
        if p0 is not None and len(p0) >= 8:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
            good = st.ravel() == 1
            if good.sum() >= 8:
                m, _ = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                                   ransacReprojThreshold=2.0)
                if m is not None:
                    step[:2] = m
        to_cur = step @ to_cur
        maps.append(np.linalg.inv(to_cur)[:2].copy())
        prev = cur
    cap.release()
    return np.stack(maps)


def _h(m: np.ndarray) -> np.ndarray:
    return np.vstack([m, [0.0, 0.0, 1.0]])


def rereference(maps: np.ndarray, ref: int) -> np.ndarray:
    """Make frame `ref` the reference instead of frame 0.

    maps[i] takes frame-i pixels to frame-0 pixels and is a chain of frame-to-frame
    estimates. inv(maps[ref]) @ maps[i] takes frame i to frame ref, and for i > ref it
    depends only on the steps between ref and i. So a hand waving in front of the camera
    before `ref` (which corrupts those early steps) no longer affects the measurement.
    """
    ref = int(min(max(ref, 0), len(maps) - 1))
    inv_ref = np.linalg.inv(_h(maps[ref]))
    return np.stack([(inv_ref @ _h(m))[:2] for m in maps])


def motion_summary(maps: np.ndarray, start: int = 0, end: int | None = None) -> dict:
    """How much the camera moved over frames [start, end), in px and degrees."""
    maps = maps[start:end]
    fwd = np.stack([np.linalg.inv(_h(m))[:2] for m in maps])
    ang = np.degrees(np.arctan2(fwd[:, 1, 0], fwd[:, 0, 0]))
    return {"shift_x_px": float(np.ptp(fwd[:, 0, 2])), "shift_y_px": float(np.ptp(fwd[:, 1, 2])),
            "rotation_deg": float(np.ptp(ang))}


def apply_to_track(track: Track, maps: np.ndarray) -> Track:
    """Map every tracked point into the reference frame's coordinates."""
    n = min(len(track.t), len(maps))
    x, y = track.x.copy(), track.y.copy()
    m = maps[:n]
    x[:n] = m[:, 0, 0] * track.x[:n] + m[:, 0, 1] * track.y[:n] + m[:, 0, 2]
    y[:n] = m[:, 1, 0] * track.x[:n] + m[:, 1, 1] * track.y[:n] + m[:, 1, 2]
    out = Track(track.t, x, y, track.area, track.conf, track.source, dict(track.meta))
    out.meta["stabilized"] = True
    return out


def load_or_estimate(video_path: str | Path, cache: str | Path, refresh: bool = False) -> np.ndarray:
    cache = Path(cache)
    if refresh or not cache.exists():
        np.save(cache, estimate_camera_motion(video_path))
    return np.load(cache)
