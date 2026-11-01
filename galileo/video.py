"""Video preprocessing with ffmpeg and exact per-frame timestamps.

Why preprocess at all:
* Phones store a rotation flag; ffmpeg applies it when re-encoding, so the
  frames we analyse and the frames SAM 3.1 sees have the same orientation.
* Capping the long side at 1280 px keeps uploads small; a mask centroid is
  sub-pixel accurate anyway.
* `-fps_mode passthrough` keeps the phone's real frame timestamps. Android
  phones often record variable frame rate, so `t = i / fps` can be wrong by
  several milliseconds per frame. We read every frame's true timestamp instead.
* `+faststart` puts the MP4 index first, which the Meta Files API prefers.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary. imageio-ffmpeg ships one, so no system install is needed."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def preprocess(src: str | Path, dst: str | Path, max_side: int = 1280) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"video not found: {src.resolve()}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"scale='min({max_side},iw)':'min({max_side},ih)'"
          ":force_original_aspect_ratio=decrease:force_divisible_by=2")
    base = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-vf", vf]
    tail = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(base + ["-fps_mode", "passthrough"] + tail, check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as first:
        # ffmpeg < 5.1 has no -fps_mode; -vsync 0 is the old spelling of passthrough.
        try:
            subprocess.run(base + ["-vsync", "0"] + tail, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            raise RuntimeError(f"ffmpeg could not convert {src}:\n{first.stderr.strip()}") from None
    return dst


_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")


def frame_times(path: str | Path) -> np.ndarray:
    """Presentation time (s) of every decoded frame, starting at 0."""
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path), "-vf", "showinfo", "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    pairs = [(int(m.group(1)), float(m.group(3))) for m in _SHOWINFO.finditer(proc.stderr)]
    if not pairs:
        raise RuntimeError("ffmpeg showinfo produced no frames; is this a video file?")
    pairs.sort()
    t = np.array([p[1] for p in pairs])
    return t - t[0]


def video_size(path: str | Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def read_frames(path: str | Path, indices) -> dict[int, np.ndarray]:
    """Several frames in one sequential pass (grab() skips decoding the frames in between)."""
    import cv2

    want = sorted({int(i) for i in indices})
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    for target in want:
        while i < target:
            if not cap.grab():
                break
            i += 1
        ok, frame = cap.read()
        if not ok:
            break
        out[target] = frame
        i += 1
    cap.release()
    return out


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """One BGR frame by index (decoding sequentially, which is exact for any frame rate)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    frame = None
    for _ in range(index + 1):
        ok, frame = cap.read()
        if not ok:
            break
    cap.release()
    if frame is None:
        raise RuntimeError(f"could not read frame {index} of {path}")
    return frame


def prepare(src: str | Path, out_dir: str | Path, refresh: bool = False) -> tuple[Path, np.ndarray, dict]:
    """Preprocess once and cache: returns (video path, frame times, info)."""
    out_dir = Path(out_dir)
    vid, times_path = out_dir / "video.mp4", out_dir / "frame_times.json"
    if refresh or not vid.exists():
        preprocess(src, vid)
    if refresh or not times_path.exists():
        times_path.write_text(json.dumps(frame_times(vid).round(6).tolist()))
    t = np.asarray(json.loads(times_path.read_text()))
    w, h = video_size(vid)
    dt = np.diff(t)
    info = {"width": w, "height": h, "frames": int(len(t)),
            "duration_s": float(t[-1]) if len(t) else 0.0,
            "fps_mean": float(1 / dt.mean()) if len(dt) else 0.0,
            "frame_interval_jitter_ms": float(dt.std() * 1e3) if len(dt) else 0.0}
    return vid, t, info