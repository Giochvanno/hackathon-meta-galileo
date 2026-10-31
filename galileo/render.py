"""Render what the measurement saw: SAM 3.1's mask, the trail, and the numbers, on every frame.

    python -m galileo.cli render runs/experiment01

Everything drawn is real output of the run: the masks come from the cached SAM answer,
the live period from the measured equilibrium crossings, the final numbers from
result.json. Nothing is re-fitted or smoothed for looks.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

MASK_BGR = (255, 200, 60)      # cyan-blue fill
TRAIL_BGR = (40, 110, 255)     # orange trail
TEXT_BGR = (255, 255, 255)
PANEL_BGR = (20, 20, 20)


def _sam_file(run: Path, meta: dict) -> Path:
    raw = meta.get("raw_output")
    if raw:
        cand = Path(str(raw).replace("\\", "/"))
        if cand.exists():
            return cand
        if (run / cand.name).exists():
            return run / cand.name
    files = sorted(run.glob("sam_*.txt"), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no sam_*.txt in {run}: render needs a SAM run")
    return files[-1]


def _blend(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color, alpha: float) -> None:
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, thick: int = 2) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, TEXT_BGR, thick, cv2.LINE_AA)


def _draw_mask(img: np.ndarray, left: int, top: int, raster: np.ndarray) -> None:
    h, w = raster.shape
    x1, y1 = min(left + w, img.shape[1]), min(top + h, img.shape[0])
    m = raster[: y1 - top, : x1 - left].astype(bool)
    roi = img[top:y1, left:x1]
    roi[m] = (roi[m] * 0.45 + np.array(MASK_BGR) * 0.55).astype(np.uint8)
    contours, _ = cv2.findContours(raster.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    for c in contours:
        cv2.drawContours(img, [c + np.array([left, top])], -1, MASK_BGR, 2, cv2.LINE_AA)


def _running_period(crossings: list[float], t_now: float) -> tuple[float | None, int]:
    """Period from the equilibrium crossings seen so far (crossing k at t0 + k T / 2)."""
    tc = [c for c in crossings if c <= t_now]
    if len(tc) < 3:
        return None, len(tc)
    return 2 * (tc[-1] - tc[0]) / (len(tc) - 1), len(tc)


def render_overlay(run_dir: str | Path, out: str | Path | None = None,
                   trail_s: float = 1.5) -> Path:
    from .sam_track import masks_by_frame, parse_sam_output

    run = Path(run_dir)
    payload = json.loads((run / "result.json").read_text())
    res, meta, info = payload["result"], payload.get("track_meta", {}), payload.get("video", {})
    t = np.asarray(json.loads((run / "frame_times.json").read_text()))
    sam_text = _sam_file(run, meta).read_text()
    oid = meta.get("object_id")
    masks = masks_by_frame(sam_text, oid)
    dets = [d for d in parse_sam_output(sam_text)[0] if oid is None or d.object_id == oid]
    centre = {d.frame: (d.cx, d.cy) for d in dets}

    # The pivot lives in the reference frame's coordinates; map it into each raw frame.
    pivot = res.get("pivot_px") if res.get("amplitude_source") == "arc fit" else None
    inv_maps = None
    if pivot is not None and (run / "camera_motion.npy").exists():
        from .stabilize import rereference

        maps = rereference(np.load(run / "camera_motion.npy"), int(info.get("reference_frame", 0)))
        inv_maps = [np.linalg.inv(np.vstack([m, [0, 0, 1]])) for m in maps]

    crossings = (res.get("crossings") or {}).get("crossing_times", [])
    out = Path(out or run / "overlay.mp4")
    cap = cv2.VideoCapture(str(run / "video.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(len(t) / t[-1]) if len(t) > 1 and t[-1] > 0 else 30.0
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(str(out), (w, h), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=8, macro_block_size=2)
    writer.send(None)
    n_trail = max(2, int(trail_s * fps))
    s = w / 720  # text scale relative to a 720-px-wide frame
    try:
        for i in range(len(t)):
            ok, frame = cap.read()
            if not ok:
                break
            img = frame.copy()
            if pivot is not None and i in centre:
                p = np.array([pivot[0], pivot[1], 1.0])
                if inv_maps is not None and i < len(inv_maps):
                    p = inv_maps[i] @ p
                cv2.line(img, (int(p[0]), int(p[1])), (int(centre[i][0]), int(centre[i][1])),
                         (230, 230, 230), 1, cv2.LINE_AA)
                cv2.drawMarker(img, (int(p[0]), int(p[1])), MASK_BGR, cv2.MARKER_CROSS,
                               int(22 * s), 2)
            trail = [centre[j] for j in range(max(0, i - n_trail), i + 1) if j in centre]
            for k in range(1, len(trail)):
                a = k / len(trail)
                cv2.line(img, (int(trail[k - 1][0]), int(trail[k - 1][1])),
                         (int(trail[k][0]), int(trail[k][1])),
                         tuple(int(c * a) for c in TRAIL_BGR), max(1, int(3 * a * s)), cv2.LINE_AA)
            if i in masks:
                _draw_mask(img, *masks[i])
                cx, cy = centre.get(i, (None, None))
                if cx is not None:
                    cv2.circle(img, (int(cx), int(cy)), max(2, int(4 * s)), TRAIL_BGR, -1,
                               cv2.LINE_AA)

            # heads-up panel
            _blend(img, 0, 0, w, int(150 * s), PANEL_BGR, 0.55)
            _text(img, "SAM 3.1 mask tracking" if i in masks else "waiting for SAM track",
                  (int(16 * s), int(36 * s)), 0.8 * s)
            _text(img, f"t = {t[i]:6.2f} s    frame {i + 1}/{len(t)}",
                  (int(16 * s), int(72 * s)), 0.65 * s)
            T_run, n = _running_period(crossings, t[i])
            if T_run is not None:
                _text(img, f"period so far: {T_run:.4f} s  ({n} crossings)",
                      (int(16 * s), int(106 * s)), 0.65 * s)
            if i >= len(t) - int(2.5 * fps):
                _text(img, f"g = {res['g']:.3f} +/- {res['sigma_g']:.3f} m/s^2   "
                      f"T0 = {res['T0']:.4f} s", (int(16 * s), int(140 * s)), 0.65 * s)
            writer.send(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
    finally:
        cap.release()
        writer.close()
    return out