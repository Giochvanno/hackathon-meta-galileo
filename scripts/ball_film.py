"""A picture of the bounces made from the video's own pixels (no drawing of the ball).

Each column of the picture is one frame: the strip of the frame around the tracked ball,
squeezed sideways, frames laid left to right. The ball's real images trace the flights;
the impacts found by the fit are marked under them. Free, no API: needs the prepared
video (runs/<name>/video.mp4), its track.csv and result_bounce.json.

    python scripts/ball_film.py runs/NAME [--t-start 3.2 --t-end 8] [--out film.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def ball_film(video: Path, track_csv: Path, result_json: Path, out: Path,
              t_start: float | None = None, t_end: float | None = None, col_px: int = 9) -> Path:
    tr = pd.read_csv(track_csv)
    res = json.loads(result_json.read_text())["result"]
    impacts = list(res.get("impacts_s") or [])
    ok = tr.dropna(subset=["x", "y"])
    t0 = t_start if t_start is not None else max(float(ok.t.min()) - 0.1, 0.0)
    t1 = t_end if t_end is not None else (impacts[-1] + 0.35 if impacts else float(ok.t.max()))
    rows = tr[(tr.t >= t0) & (tr.t <= t1)]
    r = float(np.sqrt(np.nanmedian(ok.area) / np.pi))
    half = int(round(1.25 * r))
    ys = ok[(ok.t >= t0) & (ok.t <= t1)].y
    y_lo, y_hi = int(max(ys.min() - 3 * r, 0)), int(ys.max() + 3 * r)
    x_fill = float(np.nanmedian(ok.x))

    cap = cv2.VideoCapture(str(video))
    cols, times, x_prev = [], [], x_fill
    for i in range(int(tr.frame.max()) + 1):
        good, frame = cap.read()
        if not good:
            break
        if i not in rows.frame.values:
            continue
        row = rows[rows.frame == i].iloc[0]
        x = row.x if np.isfinite(row.x) else x_prev
        x_prev = x
        xa = int(np.clip(round(x) - half, 0, frame.shape[1] - 2 * half))
        strip = frame[y_lo:min(y_hi, frame.shape[0]), xa:xa + 2 * half]
        cols.append(cv2.resize(strip, (col_px, strip.shape[0]), interpolation=cv2.INTER_AREA))
        times.append(float(row.t))
    cap.release()
    img = np.hstack(cols)
    h, w = img.shape[:2]
    pad = 34
    canvas = np.full((h + pad, w, 3), 250, np.uint8)
    canvas[:h] = img
    times = np.asarray(times)
    for k, ti in enumerate(impacts):
        if times[0] <= ti <= times[-1]:
            x = int(np.interp(ti, times, np.arange(len(times)) * col_px + col_px / 2))
            cv2.line(canvas, (x, h - 6), (x, h + 10), (40, 90, 230), 2)
            cv2.putText(canvas, str(k + 1), (x - 5, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (40, 90, 230), 1, cv2.LINE_AA)
    # one-second ticks along the top edge
    for s in np.arange(np.ceil(times[0]), times[-1], 1.0):
        x = int(np.interp(s, times, np.arange(len(times)) * col_px + col_px / 2))
        cv2.line(canvas, (x, 0), (x, 8), (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--video", help="default: RUN_DIR/video.mp4")
    ap.add_argument("--track", help="default: RUN_DIR/track.csv")
    ap.add_argument("--result", help="default: RUN_DIR/result_bounce.json")
    ap.add_argument("--t-start", type=float)
    ap.add_argument("--t-end", type=float)
    ap.add_argument("--out", help="default: RUN_DIR/film.png")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = ball_film(Path(a.video or run / "video.mp4"), Path(a.track or run / "track.csv"),
                    Path(a.result or run / "result_bounce.json"), Path(a.out or run / "film.png"),
                    a.t_start, a.t_end)
    print(out)


if __name__ == "__main__":
    main()
