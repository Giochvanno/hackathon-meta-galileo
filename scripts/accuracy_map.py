"""Accuracy map: how well does the instrument measure g, and where does it break?

Renders synthetic pendulum videos with a known g, changing one condition at a time
from a baseline (blur, noise, bob size, contrast, frame rate, duration, camera shake,
rolling shutter, swing angle, tilt, damping), runs the normal pipeline on each
(`galileo.cli pendulum`, the same code as for real videos), and compares the result
with the truth.

Two numbers per run matter:
  error  = (g_measured - g_true) / g_true        how far off
  pull   = (g_measured - g_true) / sigma_g       whether the error bar is honest
           (over many runs, pulls should scatter with a standard deviation of ~1)

The length is given exactly (its error is set to 0.1 mm), so this measures the
video part of the instrument. Results go to OUT/results.csv, OUT/summary.md and
OUT/accuracy_map.png. Finished runs are skipped when the script is run again.

    python scripts/accuracy_map.py --tracker opencv --out runs/accuracy_cv     # free
    python scripts/accuracy_map.py --tracker sam    --out runs/accuracy_sam    # ~ $8 for 81 videos
    python scripts/accuracy_map.py --tracker sam --quick                       # 1 seed: ~ $3
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from galileo.synth import Scene, render  # noqa: E402

BASE = Scene()
# factor -> (Scene field, levels, unit, label). The baseline value is always included.
FACTORS = {
    "motion blur": ("shutter", [0.0, 0.5, 1.0], "of frame", "shutter open"),
    "sensor noise": ("noise", [3.0, 10.0, 25.0], "grey levels", "noise σ"),
    "bob size": ("bob_radius_px", [5.0, 10.0, 20.0, 40.0], "px", "bob radius"),
    "contrast": ("contrast", [120.0, 40.0, 15.0], "grey levels", "bob vs background"),
    "frame rate": ("fps", [15.0, 30.0, 60.0], "fps", "frame rate"),
    "duration": ("seconds", [5.0, 10.0, 15.0, 30.0], "s", "recording"),
    "camera shake": ("shake_px", [0.0, 3.0, 10.0, 25.0], "px", "shake"),
    "rolling shutter": ("rolling_shutter_ms", [0.0, 15.0, 33.0], "ms", "read-out"),
    "swing angle": ("theta0_deg", [5.0, 12.0, 25.0, 40.0], "°", "release angle"),
    "phone tilt": ("tilt_deg", [0.0, 8.0, 20.0], "°", "roll"),
    "damping": ("gamma", [0.02, 0.1, 0.3], "1/s", "damping rate"),
}


def plan(seeds: int) -> list[tuple[str, float, int, Scene]]:
    runs, seen = [], set()
    for factor, (field, levels, *_rest) in FACTORS.items():
        for level in levels:
            for seed in range(seeds):
                scene = replace(BASE, **{field: level}, seed=seed)
                key = (field, level, seed) if level != getattr(BASE, field) else ("base", 0, seed)
                name = "base" if key[0] == "base" else f"{field}={level:g}"
                if key in seen:
                    runs.append((factor, level, seed, scene, f"{name}_s{seed}", True))
                    continue
                seen.add(key)
                runs.append((factor, level, seed, scene, f"{name}_s{seed}", False))
    return runs


def run_one(name: str, scene: Scene, out: Path, tracker: str) -> dict:
    video = out / "videos" / f"{name}.mp4"
    run_dir = out / "runs" / name
    result = run_dir / "result.json"
    if not result.exists():
        if not video.with_suffix(".truth.json").exists():  # written last: video is complete
            render(scene, video)
        truth = json.loads(video.with_suffix(".truth.json").read_text())
        x, y = truth["bob_xy"][0]
        cmd = [sys.executable, "-m", "galileo.cli", "pendulum", str(video), "--tracker", tracker,
               "--length", str(scene.length_m), "--length-err", "0.0001", "--out", str(run_dir)]
        if tracker == "sam":
            cmd += ["--point", f"{x:.0f},{y:.0f}", "--prompt-frame", "0"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if not result.exists():
            return {"name": name, "ok": False, "error": (p.stderr or p.stdout)[-300:]}
    r = json.loads(result.read_text())["result"]
    err = (r["g"] - scene.g) / scene.g
    return {"name": name, "ok": True, "g": r["g"], "sigma_g": r["sigma_g"],
            "error_pct": 100 * err, "pull": (r["g"] - scene.g) / r["sigma_g"],
            "reliable": r["reliable"], "problems": "; ".join(r["problems"]),
            "best_model": r["best_model"], "amplitude_deg": r["amplitude_deg"]}


def summarize(rows: list[dict], out: Path) -> None:
    lines = ["# Accuracy map (synthetic videos, known g)", "",
             "Error = measured g vs true g. Pull = error / reported σ (honest error bars: "
             "spread ≈ 1). The length is exact here, so this is the video part of the error.", "",
             "| Condition | Value | Runs | Median abs error | Worst abs error | Median σ_g / g | "
             "Pull spread | Flagged unreliable |", "|---|---|---|---|---|---|---|---|"]
    for factor, (field, levels, unit, label) in FACTORS.items():
        for level in levels:
            sel = [r for r in rows if r["factor"] == factor and r["level"] == level and r["ok"]]
            bad = [r for r in rows if r["factor"] == factor and r["level"] == level and not r["ok"]]
            if not sel and not bad:
                continue
            base = " (baseline)" if level == getattr(BASE, field) else ""
            if sel:
                e = np.abs([r["error_pct"] for r in sel])
                rel = np.array([r["sigma_g"] / r["g"] * 100 for r in sel])
                pulls = np.array([r["pull"] for r in sel])
                spread = float(np.sqrt(np.mean(pulls**2)))
                unrel = sum(not r["reliable"] for r in sel)
                lines.append(f"| {factor} | {level:g} {unit}{base} | {len(sel)} | {np.median(e):.3f} % | "
                             f"{e.max():.3f} % | {np.median(rel):.3f} % | {spread:.1f} | "
                             f"{unrel}/{len(sel)}{' + ' + str(len(bad)) + ' failed' if bad else ''} |")
            else:
                lines.append(f"| {factor} | {level:g} {unit}{base} | 0 | — | — | — | — | "
                             f"{len(bad)} failed |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(FACTORS)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(13, 9), facecolor="#fcfcfb")
    for ax, (factor, (field, levels, unit, label)) in zip(axes.ravel(), FACTORS.items()):
        ax.set_facecolor("#fcfcfb")
        for i, level in enumerate(levels):
            sel = [r for r in rows if r["factor"] == factor and r["level"] == level and r["ok"]]
            failed = [r for r in rows if r["factor"] == factor and r["level"] == level
                      and not r["ok"]]
            for r in sel:
                e = abs(r["error_pct"])
                off_scale = e > 20
                ax.plot(i, min(max(e, 1e-4), 20), "^" if off_scale else "o", ms=6,
                        color="#2a78d6" if r["reliable"] else "#eb6834", alpha=0.85)
                ax.plot(i, min(r["sigma_g"] / r["g"] * 100, 20), "_", ms=12, color="#52514e")
            if failed:
                ax.plot(i, 20, "x", ms=8, color="#52514e")
                ax.annotate(f"{len(failed)} no track", (i, 20), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=7, color="#52514e")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 30)
        ax.set_xticks(range(len(levels)), [f"{v:g}" for v in levels])
        ax.set_title(f"{factor} ({unit})", fontsize=10)
        ax.grid(True, color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].set_ylabel("|g error|, %   (dash: reported σ)")
    fig.suptitle("Accuracy map: synthetic videos with known g  ·  blue = reliable, orange = flagged "
                 "unreliable, ▲ = off the scale, × = no track", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "accuracy_map.png", dpi=130, facecolor="#fcfcfb")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", choices=["opencv", "sam"], default="opencv")
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="one seed per condition")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    seeds = 1 if a.quick else a.seeds
    out = Path(a.out or f"runs/accuracy_{a.tracker}")
    out.mkdir(parents=True, exist_ok=True)
    runs = plan(seeds)
    unique = {name: scene for _, _, _, scene, name, dup in runs if not dup}
    frames = sum(int(s.seconds * s.fps) for s in unique.values())
    print(f"{len(unique)} videos, {frames} frames"
          + (f", SAM about ${frames * 0.2 / 1000:.2f}" if a.tracker == "sam" else ""))
    workers = a.workers or (4 if a.tracker == "sam" else 2)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(run_one, name, scene, out, a.tracker): name
                for name, scene in unique.items()}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - one broken run must not stop the map
                r = {"name": futs[f], "ok": False, "error": f"{type(e).__name__}: {e}"}
            results[r["name"]] = r
            msg = (f"g err {r['error_pct']:+.3f} %, pull {r['pull']:+.1f}"
                   f"{'' if r['reliable'] else ', UNRELIABLE'}") if r["ok"] else "FAILED " + r["error"]
            print(f"[{k}/{len(unique)}] {r['name']}: {msg}", flush=True)
    rows = []
    for factor, level, seed, scene, name, _ in runs:
        rows.append({"factor": factor, "level": level, "seed": seed, **results[name]})
    with (out / "results.csv").open("w", newline="") as f:
        keys = ["factor", "level", "seed", "name", "ok", "g", "sigma_g", "error_pct", "pull",
                "reliable", "best_model", "amplitude_deg", "problems", "error"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summarize(rows, out)
    plot(rows, out)
    print(f"\n{out / 'summary.md'}\n{out / 'accuracy_map.png'}\n{out / 'results.csv'}")


if __name__ == "__main__":
    main()