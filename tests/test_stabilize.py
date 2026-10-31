import cv2
import numpy as np

from galileo.stabilize import apply_to_track, estimate_camera_motion, motion_summary
from galileo.track import Track


def test_recovers_known_camera_shake(tmp_path):
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (520, 680, 3), dtype=np.uint8), (5, 5), 0)
    shifts = [(0, 0), (3, -2), (6, 1), (-4, 5), (2, 8), (-7, -3)]
    path = tmp_path / "shake.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for dx, dy in shifts:  # camera moves -> the scene appears shifted by (dx, dy)
        w.write(scene[20 - dy:500 - dy, 20 - dx:660 - dx].copy())
    w.release()

    maps = estimate_camera_motion(path)
    # A point fixed in the scene, seen at (300 + dx, 200 + dy), must map back to (300, 200).
    n = len(shifts)
    x = np.array([300.0 + dx for dx, _ in shifts])
    y = np.array([200.0 + dy for _, dy in shifts])
    tr = apply_to_track(Track(np.arange(n) / 30, x, y, np.full(n, np.nan), np.full(n, np.nan)), maps)
    assert np.allclose(tr.x, 300, atol=0.3) and np.allclose(tr.y, 200, atol=0.3)
    assert motion_summary(maps)["shift_x_px"] > 10
