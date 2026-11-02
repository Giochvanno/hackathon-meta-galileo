"""One measurement from a video, end to end, with no clicks: the path the web app uses.

    video -> Muse Spark looks (experiment, object, release frame, setup issues)
          -> SAM 3.1 tracks the object from that frame (point prompt)
          -> physics: pendulum (period, T0, g) or bouncing ball (impacts, e)
          -> report.png / bounce_report.png, overlay.mp4 (pendulum), summary.json

It reuses the exact code paths of the command line (`look`, `pendulum --auto`,
`bounce --auto`), so the web numbers and the README numbers come from the same code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

STEPS = ["prepare", "look", "track", "physics", "render", "done"]
STEP_TEXT = {
    "prepare": "Reading the video: exact frame times, orientation",
    "look": "Muse Spark is looking at the video",
    "track": "SAM 3.1 is tracking the object frame by frame",
    "physics": "Fitting the physics with uncertainties",
    "render": "Drawing the overlay video",
    "done": "Done",
}

MAX_SECONDS = 45.0
USD_PER_SAM_FRAME = 0.20 / 1000
USD_LOOK = 0.03


def estimate_cost(frames_tracked: int) -> float:
    return USD_LOOK + max(frames_tracked, 50) * USD_PER_SAM_FRAME


def _args(video: Path, out: Path, **kw) -> SimpleNamespace:
    base = dict(video=str(video), out=str(out), refresh=False, tracker="sam", auto=False,
                point=None, box=None, prompt="", prompt_frame=0, no_stabilize=False,
                t_start=None, t_end=None, roi=None, lat=None, height=0.0,
                amplitude_deg=None, length=None, length_err=0.003,
                drop_height=None, drop_height_err=0.01, predict_height=None,
                predict_height_err=0.005, frame_height=None, no_rolling_shutter=False)
    base.update(kw)
    return SimpleNamespace(**base)


def measure(video: str | Path, out: str | Path, *, experiment: str = "auto",
            length_m: float | None = None, length_err_m: float = 0.003,
            drop_height_m: float | None = None, lat: float | None = None,
            render: bool = True, step: Callable[[str, str], None] | None = None,
            budget_check: Callable[[float], None] | None = None) -> dict:
    """Run one measurement. `step(name, detail)` reports progress; `budget_check(usd)` may
    raise to refuse a run that would cost too much (called before any paid SAM call)."""
    from . import video as vid
    from .assistant import annotate, look_cached
    from .cli import _analyze_and_report, _track_video, bounce_from_track

    def say(name: str, detail: str = "") -> None:
        if step:
            step(name, detail)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    say("prepare")
    video_path, t, info = vid.prepare(video, out)
    if info["duration_s"] > MAX_SECONDS:
        raise ValueError(f"the video is {info['duration_s']:.0f} s long; the limit here is "
                         f"{MAX_SECONDS:.0f} s (trim it, or run the command line locally)")
    if info["frames"] < 60:
        raise ValueError("the video is too short: record at least a few seconds of motion")

    say("look", "about $0.02")
    if budget_check:
        budget_check(USD_LOOK)
    lk = look_cached(video_path, t, out)
    annotate(vid.read_frame(video_path, lk.frame_index), lk, out / "look.png")
    kind = lk.experiment if experiment == "auto" else experiment
    if kind not in ("pendulum", "bouncing ball"):
        raise ValueError("Muse Spark did not see a pendulum or a bouncing ball here. "
                         f"It says: {lk.summary}")
    if kind == "pendulum" and not length_m:
        raise ValueError("Muse Spark sees a pendulum: enter its length (pivot to the centre of "
                         "the bob) to get g. The period does not need it, g does.")

    frames_tracked = info["frames"] - lk.frame_index
    cost = estimate_cost(frames_tracked)
    if budget_check:
        budget_check(cost - USD_LOOK)
    say("track", f"{frames_tracked} frames, about ${cost - USD_LOOK:.2f}")
    a = _args(video, out, point=f"{lk.point_px[0]},{lk.point_px[1]}",
              prompt_frame=lk.frame_index, t_start=float(t[lk.frame_index]),
              lat=lat, length=length_m, length_err=length_err_m,
              drop_height=drop_height_m)
    track, out, video_path, info = _track_video(a)

    say("physics")
    summary: dict = {"experiment": kind, "look": lk.to_dict(), "video": info,
                     "cost_usd_estimate": round(cost, 3)}
    if kind == "pendulum":
        _analyze_and_report(a, track, out, video_path, info)
        r = json.loads((out / "result.json").read_text())["result"]
        summary["result"] = {k: r[k] for k in (
            "T_measured", "sigma_T_measured", "T0", "sigma_T0", "g", "sigma_g", "g_reference",
            "g_error_pct", "deviation_sigma", "amplitude_deg", "amplitude_source", "best_model",
            "cycles", "reliable", "problems", "warnings", "length_m", "sigma_length_m")}
        summary["result"]["models"] = r["models"]
        files = ["look.png", "report.png", "overlay.png"]
        if render:
            say("render")
            try:
                from .render import render_overlay

                render_overlay(out)
                files.append("overlay.mp4")
            except Exception as e:  # noqa: BLE001 - the overlay is a bonus, the numbers are not
                summary["render_error"] = str(e)
    else:
        a.frame_height = info.get("height")
        res, preds, g = bounce_from_track(a, track, out, info)
        d = res.to_dict()
        summary["result"] = {k: d[k] for k in (
            "e", "sigma_e", "best_model", "e_slope_per_s", "sigma_e_slope", "impacts_s",
            "sigma_impacts_s", "flights_s", "ratios", "sigma_ratios", "impact_speeds_m_s",
            "first_flight_s", "sigma_first_flight_s", "drop_height_from_timing_m",
            "sigma_drop_height_from_timing_m", "drop_height_from_pixels_m",
            "rolling_shutter_readout_ms", "noise_px", "reliable", "problems", "warnings",
            "models")}
        files = ["look.png", "bounce_report.png"]
    summary["files"] = [f for f in files if (out / f).exists()]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    say("done")
    return summary
