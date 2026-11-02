"""Any oscillator: a spring, a ruler on a table edge, a swing, a branch, a pendulum without a length.

What can be measured from the video alone, with no length and no scale:
  period and frequency  (two independent estimates: model fit and zero crossings)
  damping law           (none, viscous ~v, air drag ~v^2, dry friction; compared by AIC)
  damping rate, quality factor Q = pi * (number of swings for the amplitude to fall by e)

The same fits as the pendulum (pendulum.py), without the pendulum-specific steps
(arc, swing angle, g).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .pendulum import (TWO_PI, dominant_frequency, fit_models, project_principal_axis,
                       zero_crossing_period)
from .track import Track


@dataclass
class OscillationResult:
    n_points: int
    duration_s: float
    cycles: float
    best_model: str
    models: list[dict]
    period_s: float
    sigma_period_s: float
    frequency_hz: float
    sigma_frequency_hz: float
    period_crossings_s: float | None
    sigma_period_crossings_s: float | None
    consistency_z: float | None
    amplitude_px: float
    damping_per_s: float
    quality_factor: float | None
    axis: list[float]                 # direction of the motion in the image (unit vector)
    reliable: bool = True
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_oscillation(track: Track, f_min: float = 0.15, f_max: float = 12.0) -> OscillationResult:
    tr = track.finite()
    warnings: list[str] = []
    if len(tr.t) < 40:
        raise ValueError(f"only {len(tr.t)} tracked frames; need at least 40")
    if track.coverage() < 0.8:
        warnings.append(f"object found on only {track.coverage():.0%} of frames")
    t = tr.t
    s, u, _ = project_principal_axis(tr.x, tr.y)
    tau = t - 0.5 * (t[0] + t[-1])

    # The frequency search must stay below the frame rate's Nyquist limit.
    nyquist = 0.5 / float(np.median(np.diff(t)))
    f0 = dominant_frequency(t, s, f_min, min(f_max, 0.9 * nyquist))
    fits = fit_models(tau, s, TWO_PI * f0)
    if not fits:
        raise RuntimeError("no oscillation model converged; check the track")
    best = fits[0]
    r = s - best.predict(tau)
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r) < 5 * mad if mad > 0 else np.ones_like(r, dtype=bool)
    if (~keep).sum():
        warnings.append(f"{int((~keep).sum())} outlier frames removed (>5 MAD)")
        t, s, tau = t[keep], s[keep], tau[keep]
        fits = fit_models(tau, s, best.omega)
        best = fits[0]

    T = TWO_PI / best.omega
    sT = T * best.sigma_omega / best.omega
    duration = float(t[-1] - t[0])
    cycles = duration / T

    resid = s - (best.c + best.d * tau)
    late = np.abs(resid[tau > np.quantile(tau, 0.8)])
    hyst = 0.3 * float(np.percentile(late, 95)) if len(late) else 0.3 * best.amplitude()
    cross = zero_crossing_period(t, resid, hyst)
    Tc = sTc = z = None
    if cross:
        Tc, sTc = cross["T"], cross["sigma_T"]
        z = abs(T - Tc) / np.hypot(sT, sTc)
        if z > 3:
            warnings.append(f"fit and zero-crossing periods disagree ({z:.1f} sigma)")
        sT = max(sT, sTc)

    damping = best.lam
    omega0 = np.sqrt(best.omega**2 + (damping**2 if best.name == "viscous" else 0.0))
    q = float(omega0 / (2 * damping)) if damping > 0 else None

    problems = []
    snr = best.amplitude() / max(best.residual_rms, 1e-9)
    if snr < 5:
        problems.append(f"the motion ({best.amplitude():.1f} px) is not above the tracking noise "
                        f"({best.residual_rms:.1f} px): the tracked object barely moves")
    if cycles < 3:
        problems.append(f"only {cycles:.1f} cycles in the window; record at least 5")
    if cross is None:
        problems.append("too few crossings of the rest position")
    elif cross["timing_residual_ms"] > 100 * T:
        problems.append(f"the crossings are irregular ({cross['timing_residual_ms']:.0f} ms scatter)")

    models = [{"name": m.name, "aic": round(m.aic, 2), "delta_aic": round(m.aic - best.aic, 2),
               "T": TWO_PI / m.omega, "damping_per_s": m.lam, "residual_rms_px": m.residual_rms}
              for m in fits]
    return OscillationResult(
        n_points=int(len(t)), duration_s=duration, cycles=float(cycles), best_model=best.name,
        models=models, period_s=float(T), sigma_period_s=float(sT), frequency_hz=float(1 / T),
        sigma_frequency_hz=float(sT / T**2), period_crossings_s=Tc, sigma_period_crossings_s=sTc,
        consistency_z=None if z is None else float(z), amplitude_px=float(best.amplitude()),
        damping_per_s=float(damping), quality_factor=q, axis=[float(u[0]), float(u[1])],
        reliable=not problems, problems=problems, warnings=warnings)


def oscillation_report(track: Track, res: OscillationResult, path) -> None:
    """Displacement along the motion with the fitted model, and the residuals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .pendulum import fit_model
    from .report import DATA, FIT, INK, SURFACE, _style

    tr = track.finite()
    s, _, _ = project_principal_axis(tr.x, tr.y)
    tau = tr.t - 0.5 * (tr.t[0] + tr.t[-1])
    m = fit_model(res.best_model, tau, s, TWO_PI / res.period_s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for ax in (a1, a2):
        _style(ax)
    a1.plot(tr.t, s, ".", ms=3, color=DATA, label="tracked position along the motion")
    a1.plot(tr.t, m.predict(tau), color=FIT, lw=1.4, label=f"model: {res.best_model} damping")
    a1.set_ylabel("displacement, px")
    a1.legend(frameon=False, fontsize=9, loc="upper right")
    a2.plot(tr.t, s - m.predict(tau), ".", ms=2, color=DATA)
    a2.axhline(0, color=INK, lw=0.6)
    a2.set_ylabel("residual, px")
    a2.set_xlabel("time, s")
    q = f", Q ≈ {res.quality_factor:.0f}" if res.quality_factor else ""
    status = "" if res.reliable else "UNRELIABLE   "
    fig.suptitle(f"{status}period {res.period_s:.4f} ± {res.sigma_period_s:.4f} s, frequency "
                 f"{res.frequency_hz:.4f} ± {res.sigma_frequency_hz:.4f} Hz{q}",
                 color=INK, fontsize=11.5, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
