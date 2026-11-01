"""Command line: video -> SAM 3.1 track -> period -> g.

    python -m galileo.cli look     VIDEO                       # Muse Spark: what to track, from when
    python -m galileo.cli pendulum VIDEO --length 0.52 --auto  # ...and use that as the SAM prompt
    python -m galileo.cli probe    VIDEO --prompts earbud "white earbud" earphone
    python -m galileo.cli pendulum VIDEO --length 0.52 --prompt "white earbud"
    python -m galileo.cli pendulum VIDEO --length 0.52 --point 640,410   # click instead of text
    python -m galileo.cli pendulum VIDEO --length 0.52 --point 280,845 --prompt-frame 150
    python -m galileo.cli pendulum VIDEO --length 0.52 --tracker opencv  # no API, sanity check
    python -m galileo.cli analyze  runs/NAME/track.csv --length 0.52     # re-fit, no video work
    python -m galileo.cli lab add  runs/NAME/track.csv --length 0.39 --length-err 0.003 --t-start 5
    python -m galileo.cli lab show
    python -m galileo.cli render runs/NAME                           # overlay video of the SAM run
    python -m galileo.cli bounce   VIDEO --point 610,240 --prompt-frame 20 --drop-height 0.50
    python -m galileo.cli agent --goal "Measure g and test it with a prediction"
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


# ------------------------------------------------------------------- look ---
def cmd_look(a) -> None:
    """Muse Spark looks at the video: what to track, from which frame, and what is wrong."""
    from .assistant import annotate, look_cached

    out = Path(a.out or Path("runs") / Path(a.video).stem)
    video_path, t, info = vid.prepare(a.video, out)
    lk = look_cached(video_path, t, out, refresh=a.refresh)
    png = annotate(vid.read_frame(video_path, lk.frame_index), lk, out / "look.png")
    print(f"  experiment: {lk.experiment}")
    print(f"  object:     {lk.object_name}, at {lk.point_px[0]},{lk.point_px[1]} on frame "
          f"{lk.frame_index} (t = {t[lk.frame_index]:.2f} s), confidence {lk.confidence}")
    print(f"  released:   around frame {lk.release_index} (t = {lk.t_release:.2f} s)")
    if lk.pivot_px:
        print(f"  pivot:      {lk.pivot_px[0]},{lk.pivot_px[1]}")
    if lk.floor_y_px is not None:
        print(f"  floor:      y = {lk.floor_y_px}")
    for i in lk.issues:
        print(f"  ! {i['kind'].replace('_', ' ')}: {i['detail']}")
    print(f"  {lk.summary}")
    cmd = "bounce" if lk.experiment == "bouncing ball" else "pendulum"
    print(f"\n  next: python -m galileo.cli {cmd} {a.video} --point {lk.point_px[0]},{lk.point_px[1]} "
          f"--prompt-frame {lk.frame_index} --t-start {t[lk.frame_index]:.2f} ..."
          f"\n  (or simply add --auto to `{cmd}`)\n  {png}")


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


def _reference_frame(a, t: np.ndarray) -> int:
    """The frame the measurement is expressed in: the SAM prompt frame, else --t-start."""
    if getattr(a, "prompt_frame", 0):
        return int(a.prompt_frame)
    if a.t_start is not None:
        return int(min(np.searchsorted(t, a.t_start), len(t) - 1))
    return 0


def _apply_look(a, video_path: Path, t: np.ndarray, out: Path) -> None:
    """Let Muse Spark find the object and the release, and use that as the SAM prompt."""
    from .assistant import annotate, look_cached

    lk = look_cached(video_path, t, out, refresh=a.refresh)
    a.point = f"{lk.point_px[0]},{lk.point_px[1]}"
    a.prompt_frame = lk.frame_index
    if a.t_start is None:
        a.t_start = float(t[lk.frame_index])
    annotate(vid.read_frame(video_path, lk.frame_index), lk, out / "look.png")
    print(f"Muse Spark ({lk.model}): {lk.experiment}, tracking '{lk.object_name}' at "
          f"{a.point} on frame {lk.frame_index} (t = {t[lk.frame_index]:.2f} s), "
          f"confidence {lk.confidence}. {lk.summary}", file=sys.stderr)
    for i in lk.issues:
        print(f"  setup: {i['kind'].replace('_', ' ')}: {i['detail']}", file=sys.stderr)


def _track_video(a) -> tuple[Track, Path, Path, dict]:
    """Video -> (stabilized) track of the moving object. Shared by `pendulum` and `bounce`."""
    out = Path(a.out or Path("runs") / Path(a.video).stem)
    video_path, t, info = vid.prepare(a.video, out, refresh=a.refresh)
    print(f"video: {info['width']}x{info['height']}, {info['frames']} frames, "
          f"{info['fps_mean']:.2f} fps, frame-interval jitter {info['frame_interval_jitter_ms']:.2f} ms",
          file=sys.stderr)
    if getattr(a, "auto", False) and not (a.point or a.box) and a.tracker == "sam":
        _apply_look(a, video_path, t, out)
    maps = None
    if not a.no_stabilize:
        from .stabilize import load_or_estimate, motion_summary

        from .stabilize import rereference

        maps = load_or_estimate(video_path, out / "camera_motion.npy", refresh=a.refresh)
        # Reference = a frame after the release, so a hand in the first seconds is ignored.
        ref = _reference_frame(a, t)
        maps = rereference(maps, ref)
        end = int(np.searchsorted(t, a.t_end)) if a.t_end is not None else None
        info["camera_motion"] = motion_summary(maps, ref, end)
        info["reference_frame"] = ref
        cm = info["camera_motion"]
        print(f"camera moved: {cm['shift_x_px']:.1f} px x, {cm['shift_y_px']:.1f} px y, "
              f"{cm['rotation_deg']:.2f} deg after frame {ref} -> compensated", file=sys.stderr)
    if a.tracker == "opencv":
        from .cv_track import track_moving_blob

        roi = tuple(int(v) for v in a.roi.split(",")) if getattr(a, "roi", None) else None
        track = track_moving_blob(str(video_path), t, maps=maps, roi=roi)
    else:
        track = _sam_track(a, video_path, t, info, out, maps)
    track.to_csv(out / "track.csv")
    print(f"tracked on {track.coverage():.0%} of frames -> {out / 'track.csv'}", file=sys.stderr)
    return track, out, video_path, info


def cmd_pendulum(a) -> None:
    track, out, video_path, info = _track_video(a)
    _analyze_and_report(a, track, out, video_path, info)


# ------------------------------------------------------------------ bounce ---
def cmd_bounce(a) -> None:
    """Bouncing ball: coefficient of restitution from impact times (VIDEO or a track.csv)."""
    from .bounce import analyze_bounce, fit_restitution, predict_first_flight
    from .report import bounce_report

    if a.video.lower().endswith(".csv"):
        track = Track.from_csv(a.video)
        out = Path(a.out or Path(a.video).parent)
        info: dict = {}
    else:
        track, out, _, info = _track_video(a)
    g = local_g(a.lat, a.height)
    # Motion blur makes the mask of a fast ball 2-3x larger near impacts: a loose area filter.
    work = track.clean(area_tol=3.0).window(a.t_start, a.t_end)
    height = info.get("height") or a.frame_height
    res = analyze_bounce(work, a.drop_height, a.drop_height_err, g=g, frame_height_px=height,
                         rolling_shutter=not a.no_rolling_shutter)
    out.mkdir(parents=True, exist_ok=True)
    preds = []
    if res.models and a.predict_height:
        fit = fit_restitution(res.impacts_s, res.sigma_impacts_s)[0]
        for h in a.predict_height:
            T, s = predict_first_flight(h, fit, a.predict_height_err, g)
            preds.append({"drop_height_m": h, "first_flight_s": T, "sigma_s": s})
    payload = {"system": "bouncing ball", "result": res.to_dict(), "predictions": preds,
               "video": info, "tracker": track.source, "track_meta": track.meta, "g_used": g}
    (out / "result_bounce.json").write_text(json.dumps(payload, indent=2, default=float))
    bounce_report(work, res, out / "bounce_report.png")

    print()
    if not res.reliable:
        print("  !!! UNRELIABLE MEASUREMENT: " + "; ".join(res.problems))
        print("  !!! The numbers below are not a measurement. Fix the tracking first.\n")
    print(f"  {len(res.impacts_s)} impacts timed, tracking noise {res.noise_px:.2f} px")
    if res.rolling_shutter_s_per_px:
        ro = (f" (sensor readout ≈ {res.rolling_shutter_readout_ms:.0f} ms per frame)"
              if res.rolling_shutter_readout_ms else "")
        print(f"  rolling shutter corrected: {res.rolling_shutter_s_per_px * 1e6:.1f} µs per row{ro}")
    for i, (ti, si) in enumerate(zip(res.impacts_s, res.sigma_impacts_s), 1):
        flight = f"   flight {res.flights_s[i - 1]:.4f} s" if i <= len(res.flights_s) else ""
        print(f"    impact {i}: t = {ti:.4f} ± {si * 1e3:.2f} ms{flight}")
    for i, (r, s) in enumerate(zip(res.ratios, res.sigma_ratios), 2):
        print(f"    e at impact {i} (T{i}/T{i - 1}) = {r:.4f} ± {s:.4f}"
              f"   impact speed {res.impact_speeds_m_s[i - 2]:.2f} m/s")
    print(f"  coefficient of restitution e = {res.e:.4f} ± {res.sigma_e:.4f}  ({res.best_model})")
    if res.best_model == "speed-dependent":
        print(f"  e changes with the previous flight time: {res.e_slope_per_s:+.3f} ± "
              f"{res.sigma_e_slope:.3f} per second")
    print("  models by AIC: " + ", ".join(f"{m['name']} ΔAIC={m['delta_aic']:.1f}" for m in res.models))
    if res.g_from_scale:
        print(f"  cross-check from the drop height scale: g = {res.g_from_scale:.2f} ± "
              f"{res.sigma_g_from_scale:.2f} m/s² (reference {g:.4f})")
    if res.first_flight_s:
        print(f"  first flight after the drop: {res.first_flight_s:.4f} ± "
              f"{res.sigma_first_flight_s:.4f} s (from {res.first_flight_method})")
    if res.drop_height_from_timing_m:
        px = (f", from pixels {res.drop_height_from_pixels_m * 100:.1f} cm"
              if res.drop_height_from_pixels_m else "")
        print(f"  how far the ball fell: from timing {res.drop_height_from_timing_m * 100:.1f} ± "
              f"{res.sigma_drop_height_from_timing_m * 100:.1f} cm{px}")
    if res.predicted_first_flight_s and res.first_flight_s:
        print(f"  first flight expected from e and the drop height: "
              f"{res.predicted_first_flight_s:.4f} ± {res.sigma_predicted_first_flight_s:.4f} s")
    if res.bounces_end_after_s:
        print(f"  bouncing should stop {res.bounces_end_after_s:.2f} s after the first impact")
    for p_ in preds:
        print(f"  prediction: drop from {p_['drop_height_m'] * 100:.0f} cm -> first flight "
              f"{p_['first_flight_s']:.4f} ± {p_['sigma_s']:.4f} s")
    for w in res.warnings:
        print("  ! " + w)
    print(f"\n  {out / 'bounce_report.png'}\n  {out / 'result_bounce.json'}")


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
        # Stabilized positions live in the reference frame's coordinates: draw them on it.
        frame_i = info.get("reference_frame")
        if frame_i is None:
            frame_i = int(np.nanargmin(np.abs(track.t - track.t[len(track.t) // 2])))
        overlay(vid.read_frame(video_path, frame_i), work, res, out / "overlay.png")

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


# ------------------------------------------------------------------ render ---
def cmd_render(a) -> None:
    from .render import render_overlay

    out = render_overlay(a.run_dir, a.out, trail_s=a.trail)
    print(f"overlay video: {out}")


# --------------------------------------------------------------- lab, agent ---
def cmd_lab_add(a) -> None:
    from .lab import Lab

    lab = Lab(a.lab)
    eid = lab.add_experiment(a.track_csv, a.length, a.length_err, a.t_start, a.t_end, a.note,
                             amplitude_deg=a.amplitude_deg)
    r = lab.analyze(eid)
    flag = "reliable" if r.reliable else "UNRELIABLE: " + "; ".join(r.problems)
    print(f"added {eid}: L = {a.length:.3f} ± {a.length_err:.3f} m, "
          f"T0 = {r.T0:.4f} ± {r.sigma_T0:.4f} s ({flag})")


def cmd_lab_show(a) -> None:
    from .lab import Lab

    lab = Lab(a.lab)
    for e in lab.state["experiments"].values():
        r = lab.analyze(e["id"])
        print(f"{e['id']}: L = {e['length_m']:.3f} ± {e['length_err_m']:.3f} m  "
              f"T0 = {r.T0:.4f} ± {r.sigma_T0:.4f} s  g = {r.g:.3f} ± {r.sigma_g:.3f}  "
              f"{'ok' if r.reliable else 'UNRELIABLE'}  {e['note']}")
    m = lab.state["model"]
    if m:
        offset = f", length offset {m['offset_m'] * 100:+.1f} cm" if m["fit_offset"] else ""
        print(f"model: g = {m['g']:.4f} ± {m['sigma_g']:.4f} m/s² from {m['basis']}{offset}")
    for p in lab.state["predictions"].values():
        v = p["verified_by"]
        tail = (f"-> {v['verdict']} by {v['experiment']} (measured {v['T0_measured']:.4f} s, "
                f"z = {v['z']:+.2f}, {'blind' if v['blind'] else 'NOT blind'})" if v else "-> pending")
        print(f"{p['id']}: L = {p['length_m']:.3f} m, T0 = {p['T0']:.4f} ± {p['sigma_T0']:.4f} s {tail}")
    for q in lab.state["requests"]:
        done = lab.request_done_by(q)
        status = f"done by {done}" if done else "OPEN"
        print(f"request ({status}): L = {q['length_m']:.3f} m: {q['instructions']}")


def cmd_agent(a) -> None:
    from .agent import run_agent
    from .lab import Lab

    lab = Lab(a.lab)
    if not lab.state["experiments"]:
        raise RuntimeError("the lab notebook is empty: add an experiment with `lab add` first")
    out = run_agent(lab, a.goal, model=a.model, reasoning_effort=a.effort, max_turns=a.max_turns)
    if not out["final_text"]:
        print("\n(the agent ended without a final report)")


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

    lo = sub.add_parser("look", help="Muse Spark looks at the video and sets up the measurement")
    lo.add_argument("video")
    lo.add_argument("--out")
    lo.add_argument("--refresh", action="store_true", help="ask again instead of using look.json")
    lo.set_defaults(func=cmd_look)

    pe = sub.add_parser("pendulum", help="video -> track -> g")
    pe.add_argument("video")
    pe.add_argument("--tracker", choices=["sam", "opencv"], default="sam")
    pe.add_argument("--prompt", default="white earbud", help="noun phrase for SAM 3.1")
    pe.add_argument("--auto", action="store_true",
                    help="Muse Spark finds the object and the release frame (no --point needed)")
    pe.add_argument("--point", help="X,Y click on frame 0 (video px) instead of a phrase")
    pe.add_argument("--box", help="L,T,R,B box on frame 0 instead of a phrase")
    pe.add_argument("--prompt-frame", type=int, default=0,
                    help="frame the --point/--box refers to (pick one after the hand has left; "
                         "SAM tracks forward from it and bills only those frames)")
    pe.add_argument("--refresh", action="store_true", help="redo preprocessing and the SAM call")
    pe.add_argument("--roi", help="L,T,R,B: opencv tracker ignores motion outside this box")
    pe.add_argument("--no-stabilize", action="store_true",
                    help="skip camera-motion compensation (phone on a tripod)")
    fit_args(pe)
    pe.set_defaults(func=cmd_pendulum)

    bo = sub.add_parser("bounce", help="bouncing ball -> coefficient of restitution")
    bo.add_argument("video", help="a video, or a track.csv from an earlier run")
    bo.add_argument("--tracker", choices=["sam", "opencv"], default="sam")
    bo.add_argument("--prompt", default="golf ball", help="noun phrase for SAM 3.1")
    bo.add_argument("--auto", action="store_true",
                    help="Muse Spark finds the object and the release frame (no --point needed)")
    bo.add_argument("--point", help="X,Y click on the ball (video px) on --prompt-frame")
    bo.add_argument("--box", help="L,T,R,B box on --prompt-frame instead of a phrase")
    bo.add_argument("--prompt-frame", type=int, default=0,
                    help="frame for --point/--box: the ball held still just before the drop")
    bo.add_argument("--refresh", action="store_true")
    bo.add_argument("--roi", help="L,T,R,B: opencv tracker ignores motion outside this box")
    bo.add_argument("--no-stabilize", action="store_true")
    bo.add_argument("--t-start", type=float)
    bo.add_argument("--t-end", type=float)
    bo.add_argument("--no-rolling-shutter", action="store_true",
                    help="do not estimate the row-by-row exposure delay of the phone sensor")
    bo.add_argument("--frame-height", type=int, help="video height in px, for a bare track.csv")
    bo.add_argument("--drop-height", type=float,
                    help="floor to the bottom of the ball before release, m (optional)")
    bo.add_argument("--drop-height-err", type=float, default=0.005)
    bo.add_argument("--predict-height", type=float, nargs="*",
                    help="drop heights (m) to predict the first flight time for")
    bo.add_argument("--predict-height-err", type=float, default=0.005,
                    help="1σ of the drop height you will use for the predicted drop, m")
    bo.add_argument("--lat", type=float, default=35.23)
    bo.add_argument("--height", type=float, default=0.0, help="altitude, m")
    bo.add_argument("--out")
    bo.set_defaults(func=cmd_bounce)

    an = sub.add_parser("analyze", help="re-fit an existing track.csv")
    an.add_argument("track_csv")
    fit_args(an)
    an.set_defaults(func=cmd_analyze)

    rd = sub.add_parser("render", help="overlay video: SAM mask, trail and numbers on every frame")
    rd.add_argument("run_dir", help="a runs/NAME folder produced by `pendulum` with SAM")
    rd.add_argument("--out", help="default: RUN_DIR/overlay.mp4")
    rd.add_argument("--trail", type=float, default=1.5, help="trail length, s")
    rd.set_defaults(func=cmd_render)

    lb = sub.add_parser("lab", help="the lab notebook used by the agent")
    lsub = lb.add_subparsers(dest="lab_cmd", required=True)
    la = lsub.add_parser("add", help="add a measured experiment (a track.csv)")
    la.add_argument("track_csv")
    la.add_argument("--length", type=float, required=True, help="pivot to bob centre, m")
    la.add_argument("--length-err", type=float, default=0.003)
    la.add_argument("--t-start", type=float)
    la.add_argument("--t-end", type=float)
    la.add_argument("--note", default="")
    la.add_argument("--amplitude-deg", type=float,
                    help="swing angle if known (when the camera was not square to the swing)")
    la.add_argument("--lab", default="runs/lab.json")
    la.set_defaults(func=cmd_lab_add)
    ls = lsub.add_parser("show", help="print the notebook")
    ls.add_argument("--lab", default="runs/lab.json")
    ls.set_defaults(func=cmd_lab_show)

    ag = sub.add_parser("agent", help="let Muse run the experiment loop")
    ag.add_argument("--goal", default="Measure g from the experiments in the notebook, choose "
                    "the damping model, and design a falsifiable prediction for the next "
                    "experiment. If a pending prediction can be tested, test it.")
    ag.add_argument("--model", help="default: MODEL_AGENT or muse-spark-1.3")
    ag.add_argument("--effort", help="reasoning effort, default: AGENT_REASONING or high")
    ag.add_argument("--max-turns", type=int, default=16)
    ag.add_argument("--lab", default="runs/lab.json")
    ag.set_defaults(func=cmd_agent)

    a = p.parse_args(argv)
    from .sam_track import NoMovingObjectError

    try:
        a.func(a)
    except (NoMovingObjectError, FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()