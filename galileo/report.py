"""Static figures: the fit report and a trajectory overlay on a video frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pendulum import ENVELOPES, PendulumResult, fit_model, project_principal_axis
from .track import Track

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
DATA = "#8f8d87"     # measured points: neutral
FIT = "#2a78d6"      # model: series slot 1
ACCENT = "#eb6834"   # pivot / crossings: series slot 2


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fit_report(track: Track, res: PendulumResult, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = track.finite()
    s, _, _ = project_principal_axis(tr.x, tr.y)
    tau = tr.t - 0.5 * (tr.t[0] + tr.t[-1])
    best = fit_model(res.best_model, tau, s, 2 * np.pi / res.T_measured)
    tt = np.linspace(tau.min(), tau.max(), 4000)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.4), height_ratios=[2.2, 1, 1],
                             facecolor=SURFACE, constrained_layout=True)
    ax = axes[0]
    _style(ax)
    ax.plot(tr.t, s, "o", ms=2.6, color=DATA, label="measured (mask centroid)")
    ax.plot(tt + tr.t[0] - tau[0], best.predict(tt), color=FIT, lw=1.6,
            label=f"model: {res.best_model} damping")
    if best.name != "undamped":
        env = ENVELOPES[best.name](tt, best.lam) * best.amplitude()
        for sign in (1, -1):
            ax.plot(tt + tr.t[0] - tau[0], best.c + best.d * tt + sign * env, color=FIT,
                    lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("displacement along swing, px")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    title = (f"g = {res.g:.3f} ± {res.sigma_g:.3f} m/s²   ({res.g_error_pct:+.2f}% vs "
             f"{res.g_reference:.4f}, {res.deviation_sigma:+.1f}σ)")
    if not res.reliable:
        title = "UNRELIABLE: " + res.problems[0][:90]
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=24)
    amp = f"{res.amplitude_deg:.1f}° ({res.amplitude_source})" if res.amplitude_deg else "unknown"
    q = f"   Q ≈ {res.quality_factor:.0f}" if res.quality_factor else ""
    sub = (f"L = {res.length_m:.3f} ± {res.sigma_length_m:.3f} m   T₀ = {res.T0:.4f} ± "
           f"{res.sigma_T0:.4f} s   amplitude {amp}   {res.cycles:.1f} cycles, "
           f"{res.n_points} frames{q}")
    ax.text(0, 1.02, sub, transform=ax.transAxes, va="bottom", fontsize=9, color=INK_2)

    ax = axes[1]
    _style(ax)
    ax.plot(tr.t, s - best.predict(tau), "o", ms=2.2, color=DATA)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_ylabel("residual, px")
    ax.set_xlabel("time, s")

    ax = axes[2]
    _style(ax)
    if res.crossings:
        r = np.asarray(res.crossings["residuals_ms"])
        ax.plot(np.arange(len(r)), r, "o", ms=4, color=ACCENT)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_title(f"zero-crossing timing: T = {res.T_crossings:.4f} ± "
                     f"{res.sigma_T_crossings:.4f} s, scatter {res.crossings['timing_residual_ms']:.1f} ms",
                     loc="left", fontsize=9, color=INK_2)
    ax.set_ylabel("crossing residual, ms")
    ax.set_xlabel("crossing number")

    path = Path(path)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def overlay(frame_bgr: np.ndarray, track: Track, res: PendulumResult | None,
            path: str | Path) -> Path:
    """Trajectory, fitted arc and pivot drawn on a video frame (for the demo and for QA)."""
    import cv2

    img = frame_bgr.copy()
    tr = track.finite()
    pts = np.column_stack([tr.x, tr.y])
    for (x, y) in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (52, 104, 235), -1, cv2.LINE_AA)
    if res is not None and res.pivot_px is not None and res.radius_px:
        cx, cy = res.pivot_px
        cv2.circle(img, (int(cx), int(cy)), int(res.radius_px), (214, 120, 42), 1, cv2.LINE_AA)
        cv2.drawMarker(img, (int(cx), int(cy)), (214, 120, 42), cv2.MARKER_CROSS, 18, 2)
    if res is not None:
        txt = f"g = {res.g:.3f} +/- {res.sigma_g:.3f} m/s^2   T0 = {res.T0:.4f} s"
        cv2.putText(img, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    path = Path(path)
    cv2.imwrite(str(path), img)
    return path


def bounce_report(track: Track, res, path: str | Path) -> Path:
    """Two panels: height with the fitted flights and impact marks; flight times with the e law."""
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = track.finite()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for ax in (ax1, ax2):
        _style(ax)

    label = "SAM 3.1 mask centroid" if "sam" in tr.source.lower() else "tracked centroid"
    ax1.plot(tr.t, -tr.y, ".", color=DATA, ms=4, label=label)
    for k, arc in enumerate(a for a in res.arcs if a):
        tt = np.linspace(arc["t_first"], arc["t_last"], 80)
        a, b, c = arc["coef"]
        u = tt - arc["tc"]
        ax1.plot(tt, -(a * u**2 + b * u + c), color=FIT, lw=1.6,
                 label="free-flight parabola" if k == 0 else None)
    for k, (ti, si) in enumerate(zip(res.impacts_s, res.sigma_impacts_s)):
        ax1.axvline(ti, color=ACCENT, lw=0.9, alpha=0.8,
                    label="impact (parabolas meet)" if k == 0 else None)
    ax1.set_ylabel("height in image, px (up)")
    ax1.set_xlabel("time, s")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    n = np.arange(1, len(res.flights_s) + 1)
    ax2.plot(n, res.flights_s, "o", color=INK, ms=6, label="measured flight time")
    if res.models:
        best = res.models[0]
        T = [res.flights_s[0]]
        for _ in range(len(n) - 1):
            ref = best["e_is_at_flight_s"] or 0.0
            T.append(T[-1] * (best["e"] + best["e_slope_per_s"] * (T[-1] - ref)))
        ax2.plot(n, T, "-", color=FIT, lw=1.6, label=f"{best['name']} e model")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_xlabel("flight number")
    ax2.set_ylabel("flight time, s (log)")
    ax2.legend(frameon=False, fontsize=9)

    status = "" if res.reliable else "UNRELIABLE   "
    slope = (f", de/dT = {res.e_slope_per_s:+.3f} ± {res.sigma_e_slope:.3f} /s"
             if res.best_model == "speed-dependent" else "")
    fig.suptitle(f"{status}coefficient of restitution e = {res.e:.4f} ± {res.sigma_e:.4f}{slope}   "
                 f"({len(res.impacts_s)} impacts, best model: {res.best_model})",
                 color=INK, fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path