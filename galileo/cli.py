"""Command line: video -> SAM 3.1 track -> period -> g.

    python -m galileo.cli probe    VIDEO --prompts earbud "white earbud" earphone
    python -m galileo.cli pendulum VIDEO --length 0.52 --prompt "white earbud"
    python -m galileo.cli pendulum VIDEO --length 0.52 --point 640,410   # click instead of text
    python -m galileo.cli pendulum VIDEO --length 0.52 --point 280,845 --prompt-frame 150
    python -m galileo.cli pendulum VIDEO --length 0.52 --tracker opencv  # no API, sanity check
    python -m galileo.cli analyze  runs/NAME/track.csv --length 0.52     # re-fit, no video work
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import video as vid
from .pendulum import analyze_pendulum, local_g, predict_period
from .track import Track


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "prompt"


def _xy(text: str) -> tuple[int, int]:
    x, y = (int(float(v)) for v in text.split(","))
    return x, y


# ------------------------------------------------------------------ probe ---
def cmd_probe(a) -> None:
    """Try several noun phrases on one frame (about $0.0025 each) before paying for the video."""
    import cv2

    from .sam_client import get_client, segment_image
    from .sam_track import parse_sam_output

    out = Path(a.out or Path("runs") / Path(a.video).stem)
    video_path, t, info = vid.prepare(a.video, out)
    idx = int(np.searchsorted(t, a.at))
    frame = vid.read_frame(video_path, idx)
    ok, png = cv2.imencode(".png", frame)
    client = get_client()
    print(f"frame {idx} (t={t[idx]:.2f} s), {info['width']}x{info['height']}")
    for prompt in a.prompts:
        text = segment_image(client, png.tobytes(), prompt)
        dets, _ = parse_sam_output(text, media="image")
        shown = frame.copy()
        for d in dets:
            cv2.rectangle(shown, d.box[:2], (d.box[2] - 1, d.box[3] - 1), (52, 104, 235), 2)
        p = out / f"probe_{_slug(prompt)}.png"
        cv2.imwrite(str(p), shown)
        desc = ", ".join(f"id {d.object_id}: ({d.cx:.0f},{d.cy:.0f}) area {d.area} c={d.conf:.2f}"
                         for d in dets) or "no match"
        print(f"  {prompt!r:24} -> {len(dets)} match(es): {desc}   [{p}]")
    print("Pick the prompt with exactly one confident match on the bob.")


# --------------------------------------------------------------- tracking ---
def _sam_track(a, video_path: Path, t: np.ndarray, info: dict, out: Path,
               maps: np.ndarray | None = None) -> Track:
    from .sam_client import box_prompt, estimate_cost_usd, get_client, point_prompt, segment_video
    from .sam_track import choose_moving_object, parse_sam_output, tracks_from_detections

    size = (info["width"], info["height"])
    extra: dict[str, str] = {}
    f0 = a.prompt_frame
    if a.point:
        text = point_prompt(size, *_xy(a.point), frame=f0)
        tag = f"point-{a.point.replace(',', '-')}-f{f0}"
        extra["propagation_direction"] = "forward"
    elif a.box:
        l, tp, r, b = (int(v) for v in a.box.split(","))
        text = box_prompt(size, (l, tp, r, b), frame=f0)
        tag = f"box-{a.box.replace(',', '-')}-f{f0}"
        extra["propagation_direction"] = "forward"
    else:
        text, tag = a.prompt, _slug(a.prompt)

    raw_path = out / f"sam_{tag}.txt"
    if a.refresh or not raw_path.exists():
        print(f"SAM 3.1: {info['frames']} frames, about ${estimate_cost_usd(info['frames']):.3f}. "
              f"Prompt: {text[:80]!r}", file=sys.stderr)
        raw = segment_video(get_client(), video_path, text, extra_metadata=extra)
        raw_path.write_text(raw)
        print(f"raw SAM output saved: {raw_path}", file=sys.stderr)
    raw = raw_path.read_text()
    dets, warns = parse_sam_output(raw)
    for w in warns[:5]:
        print("parser:", w, file=sys.stderr)
    tracks = tracks_from_detections(dets, t)
    if maps is not None:  # judge motion in frame-0 coordinates, so camera shake doesn't count
        from .stabilize import apply_to_track

        tracks = {k: apply_to_track(v, maps) for k, v in tracks.items()}
    best, summary = choose_moving_object(tracks)
    (out / "sam_objects.json").write_text(json.dumps(summary, indent=2))
    print(f"SAM objects: {summary}\n-> using object {best}", file=sys.stderr)
    tr = tracks[best]
    tr.meta.update({"prompt": text, "raw_output": str(raw_path)})
    return tr


def cmd_pendulum(a) -> None:
    out = Path(a.out or Path("runs") / Path(a.video).stem)
    video_path, t, info = vid.prepare(a.video, out, refresh=a.refresh)
    print(f"video: {info['width']}x{info['height']}, {info['frames']} frames, "
          f"{info['fps_mean']:.2f} fps, frame-interval jitter {info['frame_interval_jitter_ms']:.2f} ms",
          file=sys.stderr)
    maps = None
    if not a.no_stabilize:
        from .stabilize import load_or_estimate, motion_summary

        maps = load_or_estimate(video_path, out / "camera_motion.npy", refresh=a.refresh)
        info["camera_motion"] = motion_summary(maps)
        cm = info["camera_motion"]
        print(f"camera moved: {cm['shift_x_px']:.1f} px x, {cm['shift_y_px']:.1f} px y, "
              f"{cm['rotation_deg']:.2f} deg -> compensated", file=sys.stderr)
    if a.tracker == "opencv":
        from .cv_track import track_moving_blob

        track = track_moving_blob(str(video_path), t, maps=maps)
    else:
        track = _sam_track(a, video_path, t, info, out, maps)
    track.to_csv(out / "track.csv")
    print(f"tracked on {track.coverage():.0%} of frames -> {out / 'track.csv'}", file=sys.stderr)
    _analyze_and_report(a, track, out, video_path, info)


def cmd_analyze(a) -> None:
    track = Track.from_csv(a.track_csv)
    out = Path(a.out or Path(a.track_csv).parent)
    video_path = out / "video.mp4"
    _analyze_and_report(a, track, out, video_path if video_path.exists() else None, {})


def _analyze_and_report(a, track: Track, out: Path, video_path: Path | None, info: dict) -> None:
    from .report import fit_report, overlay

    g_ref = local_g(a.lat, a.height) if a.lat is not None else 9.80665
    work = track.clean().window(a.t_start, a.t_end)
    res = analyze_pendulum(work, a.length, a.length_err, amplitude_deg=a.amplitude_deg,
                           g_reference=g_ref)
    cm = info.get("camera_motion")
    if cm and max(cm["shift_x_px"], cm["shift_y_px"]) > 5:
        res.warnings.append(f"camera moved {max(cm['shift_x_px'], cm['shift_y_px']):.0f} px during "
                            "the recording (compensated); fix the phone for best accuracy")
    implied_L = res.g_reference * res.T0**2 / (4 * np.pi**2)
    if abs(implied_L - a.length) / a.length > 0.1:
        res.warnings.append(f"the period fits a simple pendulum of L = {implied_L * 100:.1f} cm, "
                            f"but --length is {a.length * 100:.1f} cm: re-measure L, or the "
                            "pendulum is not simple (stiff cable, heavy cable, tumbling bob)")
    out.mkdir(parents=True, exist_ok=True)
    payload = {"result": res.to_dict(), "video": info, "tracker": track.source,
               "track_meta": track.meta}
    (out / "result.json").write_text(json.dumps(payload, indent=2, default=float))
    fit_report(work, res, out / "report.png")
    if video_path is not None:
        mid = int(np.nanargmin(np.abs(track.t - track.t[len(track.t) // 2])))
        overlay(vid.read_frame(video_path, mid), work, res, out / "overlay.png")

    print()
    if not res.reliable:
        print("  !!! UNRELIABLE MEASUREMENT: " + "; ".join(res.problems))
        print("  !!! The numbers below are not a measurement. Fix the tracking first.\n")
    print(f"  period (fit, {res.best_model}) T = {res.T_measured:.4f} ± {res.sigma_T_measured:.4f} s")
    if res.T_crossings:
        print(f"  period (zero crossings)   T = {res.T_crossings:.4f} ± {res.sigma_T_crossings:.4f} s"
              f"   agreement {res.consistency_z:.1f}σ")
    amp = f"{res.amplitude_deg:.1f}° via {res.amplitude_source}" if res.amplitude_deg else "unknown"
    print(f"  amplitude {amp}, correction ×{res.amplitude_factor:.5f}")
    print(f"  small-angle period T0 = {res.T0:.4f} ± {res.sigma_T0:.4f} s")
    print(f"  g = {res.g:.3f} ± {res.sigma_g:.3f} m/s²   "
          f"({res.g_error_pct:+.2f}% vs {res.g_reference:.4f}, {res.deviation_sigma:+.1f}σ)")
    print("  models by AIC: " + ", ".join(f"{m['name']} ΔAIC={m['delta_aic']:.1f}" for m in res.models))
    for L_new in (0.30, 0.40):
        T, s = predict_period(L_new, res.g, res.sigma_g, a.length_err)
        print(f"  prediction: L = {L_new:.2f} m -> T = {T:.3f} ± {s:.3f} s")
    for w in res.warnings:
        print("  ! " + w)
    print(f"\n  {out / 'report.png'}\n  {out / 'result.json'}")


# ------------------------------------------------------------------- main ---
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="galileo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="test noun phrases on one frame")
    pr.add_argument("video")
    pr.add_argument("--prompts", nargs="+", default=["earbud", "white earbud", "earphone"])
    pr.add_argument("--at", type=float, default=1.0, help="time of the frame to test, s")
    pr.add_argument("--out")
    pr.set_defaults(func=cmd_probe)

    def fit_args(sp):
        sp.add_argument("--length", type=float, required=True, help="pivot to bob centre, m")
        sp.add_argument("--length-err", type=float, default=0.003, help="1σ of length, m")
        sp.add_argument("--t-start", type=float, help="ignore data before this time (release)")
        sp.add_argument("--t-end", type=float)
        sp.add_argument("--amplitude-deg", type=float, help="initial swing angle if known")
        sp.add_argument("--lat", type=float, default=35.23,
                        help="latitude for reference g (Busan 35.23, Seoul 37.57)")
        sp.add_argument("--height", type=float, default=0.0, help="altitude, m")
        sp.add_argument("--out")

    pe = sub.add_parser("pendulum", help="video -> track -> g")
    pe.add_argument("video")
    pe.add_argument("--tracker", choices=["sam", "opencv"], default="sam")
    pe.add_argument("--prompt", default="white earbud", help="noun phrase for SAM 3.1")
    pe.add_argument("--point", help="X,Y click on frame 0 (video px) instead of a phrase")
    pe.add_argument("--box", help="L,T,R,B box on frame 0 instead of a phrase")
    pe.add_argument("--prompt-frame", type=int, default=0,
                    help="frame the --point/--box refers to (pick one after the hand has left; "
                         "SAM tracks forward from it and bills only those frames)")
    pe.add_argument("--refresh", action="store_true", help="redo preprocessing and the SAM call")
    pe.add_argument("--no-stabilize", action="store_true",
                    help="skip camera-motion compensation (phone on a tripod)")
    fit_args(pe)
    pe.set_defaults(func=cmd_pendulum)

    an = sub.add_parser("analyze", help="re-fit an existing track.csv")
    an.add_argument("track_csv")
    fit_args(an)
    an.set_defaults(func=cmd_analyze)

    a = p.parse_args(argv)
    from .sam_track import NoMovingObjectError

    try:
        a.func(a)
    except (NoMovingObjectError, FileNotFoundError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()