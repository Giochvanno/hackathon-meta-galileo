"""Pendulum analysis: period, damping model, g, all with uncertainties.

Pipeline (each step is a plain function so the agent can call it as a tool):
  1. project_principal_axis : 2-D track -> 1-D displacement s(t) along the swing
  2. dominant_frequency     : Lomb-Scargle on unevenly sampled data -> initial omega
  3. fit_models             : s = E(t)(a cos wt + b sin wt) + c + d t for 4 damping laws, AIC
  4. zero_crossing_period   : model-free cross-check, OLS on crossing times
  5. fit_circle             : the bob moves on an arc -> pivot, radius in px -> amplitude angle
  6. finite-amplitude and damping corrections -> small-angle period T0 -> g = 4 pi^2 L / T0^2
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lombscargle

from .track import Track

TWO_PI = 2 * np.pi
G_STANDARD = 9.80665


def local_g(lat_deg: float, height_m: float = 0.0) -> float:
    """Reference g from latitude (WGS84 normal gravity) and height (free-air)."""
    s = np.sin(np.radians(lat_deg))
    s2 = np.sin(np.radians(2 * lat_deg))
    return float(9.780327 * (1 + 0.0053024 * s**2 - 0.0000058 * s2**2) - 3.086e-6 * height_m)


# ---------------------------------------------------------------- geometry ---
def project_principal_axis(x: np.ndarray, y: np.ndarray):
    """Displacement along the main swing direction (robust to a tilted phone)."""
    p = np.column_stack([x, y])
    mu = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - mu, full_matrices=False)
    u = vt[0] if vt[0][0] >= 0 else -vt[0]
    return (p - mu) @ u, u, mu


def fit_circle(x: np.ndarray, y: np.ndarray):
    """Geometric circle fit to the bob's arc. Returns (cx, cy, R) in px.

    The arc of a 10 degree swing is very flat (sagitta ~1.5 % of R), where the
    usual algebraic (Kasa) fit underestimates R badly. We start from a parabola
    fitted in the swing frame and refine the true orthogonal distances.
    """
    from scipy.optimize import least_squares

    s, u, mu = project_principal_axis(x, y)
    v = np.array([-u[1], u[0]])
    w = (np.column_stack([x, y]) - mu) @ v
    c2, c1, c0 = np.polyfit(s, w, 2)
    if abs(c2) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    r0 = 1 / (2 * abs(c2))
    s0, w0 = -c1 / (2 * c2), c0 - c1**2 / (4 * c2)
    centre0 = mu + u * s0 + v * (w0 + np.sign(c2) * r0)
    sol = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2],
                        [centre0[0], centre0[1], r0])
    cx, cy, r = sol.x
    return float(cx), float(cy), float(abs(r))


# --------------------------------------------------------------- frequency ---
def dominant_frequency(t: np.ndarray, s: np.ndarray, f_min: float = 0.15,
                       f_max: float = 6.0) -> float:
    freqs = np.linspace(f_min, f_max, 8000)
    power = lombscargle(t, s - s.mean(), TWO_PI * freqs)
    return float(freqs[np.argmax(power)])


# ------------------------------------------------------------------ models ---
# Envelope laws E(tau), tau = t - t_mid, normalised so E(0) = 1.
#   undamped  : no energy loss
#   viscous   : F ~ v    -> exponential decay
#   quadratic : F ~ v^2  -> amplitude ~ 1/(1 + k t)   (air drag at these Reynolds numbers)
#   coulomb   : constant friction at the pivot -> linear decay
ENVELOPES = {
    "undamped": None,
    "viscous": lambda tau, lam: np.exp(-lam * tau),
    "quadratic": lambda tau, lam: 1.0 / (1.0 + lam * tau),
    "coulomb": lambda tau, lam: 1.0 - lam * tau,
}


@dataclass
class ModelFit:
    name: str
    omega: float
    sigma_omega: float
    lam: float            # damping parameter of the envelope (1/s); 0 for undamped
    sigma_lam: float
    a: float
    b: float
    c: float
    d: float
    rss: float
    n: int
    k: int
    aic: float
    residual_rms: float
    autocorr_inflation: float

    def predict(self, tau: np.ndarray) -> np.ndarray:
        env = 1.0 if self.name == "undamped" else ENVELOPES[self.name](tau, self.lam)
        return env * (self.a * np.cos(self.omega * tau) + self.b * np.sin(self.omega * tau)) \
            + self.c + self.d * tau

    def amplitude(self) -> float:
        return float(np.hypot(self.a, self.b))


def _autocorr_inflation(r: np.ndarray) -> float:
    """sqrt((1+rho)/(1-rho)) for lag-1 autocorrelation rho of the residuals.

    Least-squares errors assume independent residuals. Tracking errors are
    correlated from frame to frame, so naive errors are too small; this is the
    standard AR(1) effective-sample-size correction.
    """
    if len(r) < 10:
        return 1.0
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    rho = min(max(rho, 0.0), 0.95)
    return float(np.sqrt((1 + rho) / (1 - rho)))


def fit_model(name: str, tau: np.ndarray, s: np.ndarray, omega0: float) -> ModelFit:
    # Linear least squares at the initial frequency gives a, b, c, d for free.
    basis = np.column_stack([np.cos(omega0 * tau), np.sin(omega0 * tau), np.ones_like(tau), tau])
    (a0, b0, c0, d0), *_ = np.linalg.lstsq(basis, s, rcond=None)
    half_span = float(np.max(np.abs(tau)))

    if name == "undamped":
        def f(tt, a, b, w, c, d):
            return a * np.cos(w * tt) + b * np.sin(w * tt) + c + d * tt
        p0 = [a0, b0, omega0, c0, d0]
        lo = [-np.inf, -np.inf, 0.7 * omega0, -np.inf, -np.inf]
        hi = [np.inf, np.inf, 1.3 * omega0, np.inf, np.inf]
    else:
        env = ENVELOPES[name]

        def f(tt, a, b, w, lam, c, d):
            return env(tt, lam) * (a * np.cos(w * tt) + b * np.sin(w * tt)) + c + d * tt
        lam_hi = 5.0 if name == "viscous" else 0.95 / half_span
        p0 = [a0, b0, omega0, min(0.01, 0.5 * lam_hi), c0, d0]
        lo = [-np.inf, -np.inf, 0.7 * omega0, 0.0, -np.inf, -np.inf]
        hi = [np.inf, np.inf, 1.3 * omega0, lam_hi, np.inf, np.inf]

    popt, pcov = curve_fit(f, tau, s, p0=p0, bounds=(lo, hi), maxfev=20000)
    resid = s - f(tau, *popt)
    rss = float(resid @ resid)
    n, k = len(s), len(popt)
    infl = _autocorr_inflation(resid)
    err = np.sqrt(np.clip(np.diag(pcov), 0, None)) * infl
    if name == "undamped":
        a, b, w, c, d = popt
        lam, s_lam, s_w = 0.0, 0.0, err[2]
    else:
        a, b, w, lam, c, d = popt
        s_lam, s_w = err[3], err[2]
    return ModelFit(name, float(w), float(s_w), float(lam), float(s_lam), float(a), float(b),
                    float(c), float(d), rss, n, k, float(n * np.log(rss / n) + 2 * k),
                    float(np.sqrt(rss / n)), infl)


def fit_models(tau: np.ndarray, s: np.ndarray, omega0: float,
               names: tuple[str, ...] = ("undamped", "viscous", "quadratic", "coulomb")) -> list[ModelFit]:
    fits = []
    for name in names:
        try:
            fits.append(fit_model(name, tau, s, omega0))
        except (RuntimeError, ValueError):
            continue
    return sorted(fits, key=lambda m: m.aic)


# ----------------------------------------------------------- cross-check ---
def zero_crossing_period(t: np.ndarray, r: np.ndarray, hysteresis: float):
    """Period from equilibrium-crossing times, independent of any model.

    Crossings are linearly interpolated; hysteresis stops noise near zero from
    producing extra crossings. Crossing k happens at t0 + k T/2, so an
    ordinary least-squares line through (k, t_k) gives T and its standard error.
    """
    crossings: list[float] = []
    state, last_i = 0, 0
    for i, v in enumerate(r):
        new = 1 if v > hysteresis else (-1 if v < -hysteresis else 0)
        if new == 0:
            continue
        if state != 0 and new != state:
            seg = r[last_i:i + 1]
            idx = np.nonzero(np.sign(seg[:-1]) != np.sign(seg[1:]))[0]
            if len(idx):
                j = last_i + idx[-1]
                t_c = t[j] - r[j] * (t[j + 1] - t[j]) / (r[j + 1] - r[j])
                crossings.append(float(t_c))
        state, last_i = new, i
    tc = np.asarray(crossings)
    if len(tc) < 4:
        return None
    k = np.arange(len(tc))
    a = np.column_stack([np.ones_like(k, dtype=float), k])
    coef, *_ = np.linalg.lstsq(a, tc, rcond=None)
    res = tc - a @ coef
    dof = len(tc) - 2
    s2 = float(res @ res) / dof
    se_slope = float(np.sqrt(s2 / np.sum((k - k.mean()) ** 2)))
    return {"T": 2 * float(coef[1]), "sigma_T": 2 * se_slope, "n_crossings": int(len(tc)),
            "crossing_times": tc.tolist(), "timing_residual_ms": float(np.sqrt(s2) * 1e3),
            "residuals_ms": (res * 1e3).tolist()}


# ------------------------------------------------------------------- result ---
@dataclass
class PendulumResult:
    length_m: float
    sigma_length_m: float
    n_points: int
    duration_s: float
    cycles: float
    best_model: str
    models: list[dict]
    T_measured: float            # damped period at the actual amplitude, from the best model
    sigma_T_measured: float
    T_crossings: float | None
    sigma_T_crossings: float | None
    consistency_z: float | None  # |T_fit - T_cross| / combined sigma
    amplitude_px: float
    amplitude_deg: float | None
    amplitude_source: str
    amplitude_factor: float      # T(theta) / T(0) averaged over the window
    pivot_px: list[float] | None
    radius_px: float | None
    px_per_m: float | None
    T0: float                    # small-angle, undamped period
    sigma_T0: float
    g: float
    sigma_g: float
    g_reference: float
    g_error_pct: float
    deviation_sigma: float
    damping_per_s: float
    quality_factor: float | None
    warnings: list[str] = field(default_factory=list)
    crossings: dict | None = None
    reliable: bool = True
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def amplitude_factor(theta: np.ndarray) -> float:
    """Mean of T(theta)/T(0) = 1 + theta^2/16 + 11 theta^4/3072 over the samples."""
    return float(np.mean(1 + theta**2 / 16 + 11 * theta**4 / 3072))


def analyze_pendulum(track: Track, length_m: float, sigma_length_m: float = 0.003, *,
                     amplitude_deg: float | None = None, g_reference: float = G_STANDARD) -> PendulumResult:
    tr = track.finite()
    warnings: list[str] = []
    if len(tr.t) < 50:
        raise ValueError(f"only {len(tr.t)} tracked frames; need at least 50")
    if track.coverage() < 0.8:
        warnings.append(f"object found on only {track.coverage():.0%} of frames")

    t, x, y = tr.t, tr.x, tr.y
    s, _, _ = project_principal_axis(x, y)
    t_mid = 0.5 * (t[0] + t[-1])
    tau = t - t_mid

    f0 = dominant_frequency(t, s)
    fits = fit_models(tau, s, TWO_PI * f0)
    if not fits:
        raise RuntimeError("no model converged; check the track")
    best = fits[0]

    # One robust re-fit without outlier frames (> 5 MAD from the best model).
    r = s - best.predict(tau)
    mad = 1.4826 * np.median(np.abs(r - np.median(r)))
    keep = np.abs(r) < 5 * mad if mad > 0 else np.ones_like(r, dtype=bool)
    if (~keep).sum():
        warnings.append(f"{int((~keep).sum())} outlier frames removed (>5 MAD)")
        t, x, y, s, tau = t[keep], x[keep], y[keep], s[keep], tau[keep]
        fits = fit_models(tau, s, best.omega)
        best = fits[0]

    T_meas = TWO_PI / best.omega
    sig_T_meas = T_meas * best.sigma_omega / best.omega
    duration = float(t[-1] - t[0])
    cycles = duration / T_meas
    if cycles < 5:
        warnings.append(f"only {cycles:.1f} cycles; record at least 10 for a good period")

    # Model-free cross-check.
    resid_osc = s - (best.c + best.d * tau)
    late = np.abs(resid_osc[tau > np.quantile(tau, 0.8)])
    hyst = 0.3 * float(np.percentile(late, 95)) if len(late) else 0.3 * best.amplitude()
    cross = zero_crossing_period(t, resid_osc, hyst)
    T_cross = sig_cross = z = None
    if cross:
        T_cross, sig_cross = cross["T"], cross["sigma_T"]
        z = abs(T_meas - T_cross) / np.hypot(sig_T_meas, sig_cross)
        if z > 3:
            warnings.append(f"fit and zero-crossing periods disagree ({z:.1f} sigma)")
        sig_T_meas = max(sig_T_meas, sig_cross)  # conservative: take the larger error

    # Amplitude angle: from the arc geometry, or given by the user.
    env = (np.ones_like(tau) if best.name == "undamped" else ENVELOPES[best.name](tau, best.lam))
    amp_t = best.amplitude() * np.abs(env)  # chord half-width over time, px
    pivot = radius = px_per_m = None
    theta_amp_deg = None
    source = "none"
    cx, cy, R = fit_circle(x, y)
    arc_ok = np.isfinite(R) and 1.5 * best.amplitude() < R < 40 * best.amplitude() and cy < np.mean(y)
    if amplitude_deg is not None:
        theta_t = np.radians(amplitude_deg) * np.abs(env) / np.max(np.abs(env))
        source = "user"
    elif arc_ok:
        theta_t = np.arcsin(np.clip(amp_t / R, 0, 1))
        pivot, radius, px_per_m = [float(cx), float(cy)], float(R), float(R / length_m)
        source = "arc fit"
    else:
        theta_t = np.zeros_like(tau)
        warnings.append("arc too flat to estimate the swing angle; no finite-amplitude "
                        "correction applied (keep the pivot in frame, or pass --amplitude-deg)")
    if source != "none":
        theta_amp_deg = float(np.degrees(np.max(theta_t)))
    k_amp = amplitude_factor(theta_t)

    # Small-angle, undamped period. Viscous damping shifts omega: w0^2 = wd^2 + gamma^2.
    gamma = best.lam if best.name == "viscous" else 0.0
    omega0 = np.sqrt(best.omega**2 + gamma**2)
    T0 = TWO_PI / omega0 / k_amp
    sig_T0 = sig_T_meas / k_amp
    if source == "none":  # unknown amplitude: count a typical 10 degree correction as systematic
        sig_T0 = float(np.hypot(sig_T0, T0 * (np.radians(10) ** 2 / 16)))

    g = 4 * np.pi**2 * length_m / T0**2
    sig_g = g * np.hypot(sigma_length_m / length_m, 2 * sig_T0 / T0)

    # Effective damping rate (1/s) for any envelope: -d ln E / dt at the middle.
    damping = best.lam
    q = float(omega0 / (2 * damping)) if damping > 0 else None

    models = [{"name": m.name, "aic": round(m.aic, 2), "delta_aic": round(m.aic - best.aic, 2),
               "T": TWO_PI / m.omega, "sigma_T": TWO_PI / m.omega * m.sigma_omega / m.omega,
               "damping_per_s": m.lam, "residual_rms_px": m.residual_rms,
               "autocorr_inflation": m.autocorr_inflation} for m in fits]

    # Is there a pendulum in this track at all? Signal-to-noise and timing regularity.
    problems = []
    snr = best.amplitude() / max(best.residual_rms, 1e-9)
    if snr < 5:
        problems.append(f"swing amplitude {best.amplitude():.1f} px is not above the tracking noise "
                        f"{best.residual_rms:.1f} px (the tracked object barely moves)")
    if cycles < 3:
        problems.append(f"only {cycles:.1f} cycles in the window")
    if cross is None:
        problems.append("too few equilibrium crossings")
    elif cross["timing_residual_ms"] > 100 * T_meas:  # scatter > 10 % of the period
        problems.append(f"equilibrium crossings are irregular ({cross['timing_residual_ms']:.0f} ms "
                        "scatter)")

    return PendulumResult(
        reliable=not problems, problems=problems,
        length_m=length_m, sigma_length_m=sigma_length_m, n_points=int(len(t)),
        duration_s=duration, cycles=cycles, best_model=best.name, models=models,
        T_measured=T_meas, sigma_T_measured=sig_T_meas, T_crossings=T_cross,
        sigma_T_crossings=sig_cross, consistency_z=None if z is None else float(z),
        amplitude_px=best.amplitude(), amplitude_deg=theta_amp_deg, amplitude_source=source,
        amplitude_factor=k_amp, pivot_px=pivot, radius_px=radius, px_per_m=px_per_m,
        T0=float(T0), sigma_T0=float(sig_T0), g=float(g), sigma_g=float(sig_g),
        g_reference=g_reference, g_error_pct=float(100 * (g - g_reference) / g_reference),
        deviation_sigma=float((g - g_reference) / sig_g), damping_per_s=float(damping),
        quality_factor=q, warnings=warnings, crossings=cross)


def predict_period(length_m: float, g: float, sigma_g: float, sigma_length_m: float = 0.003,
                   amplitude_deg: float = 0.0) -> tuple[float, float]:
    """Period for a new length, with 1-sigma uncertainty. The agent's `predict` tool."""
    th = np.radians(amplitude_deg)
    k = 1 + th**2 / 16 + 11 * th**4 / 3072
    T = TWO_PI * np.sqrt(length_m / g) * k
    sig = 0.5 * T * np.hypot(sigma_length_m / length_m, sigma_g / g)
    return float(T), float(sig)
