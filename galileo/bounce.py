"""Bouncing ball: impact times between frames, coefficient of restitution, predictions.

The coefficient of restitution e is the ratio of speeds just after and just before
an impact. Between two impacts the ball is in free flight, and a free flight that
starts upward at speed v lasts T = 2 v / g. So

    e = v_after / v_before = T_next / T_previous

and the measurement needs only *times*: no ruler, no pixel-to-metre scale, the same
reason the pendulum needs only the period. Phone clocks are precise; distances on
a phone video are not.

Pipeline (plain functions, so an agent can call them as tools):
  1. find_impacts     : lowest points of the trajectory (largest image y)
  2. fit_arcs         : one parabola per flight, noise pooled over all flights
  3. impact_times     : where neighbouring parabolas meet -> impact time between frames
  4. fit_restitution  : weighted fit to all impact times; constant e versus e that
                        depends on the impact speed, compared by AIC
  5. predictions      : later impacts, first flight after a drop from a new height
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .track import Track

G_BUSAN = 9.7975
MIN_ARC_POINTS = 5   # a parabola has 3 parameters; keep 2 degrees of freedom


# ------------------------------------------------------------------ noise ---
def noise_px(y: np.ndarray) -> float:
    """Tracking noise from third differences (zero for a parabola, 20 sigma^2 for noise)."""
    d3 = np.diff(y, 3)
    if len(d3) < 5:
        return float("nan")
    mad = np.median(np.abs(d3 - np.median(d3)))
    return float(1.4826 * mad / np.sqrt(20))


# ---------------------------------------------------------------- impacts ---
def find_impacts(y: np.ndarray, noise: float, min_prominence_px: float = 3.0) -> np.ndarray:
    """Indices of the frames closest to each impact: local maxima of image y."""
    prom = max(min_prominence_px, 8 * noise if np.isfinite(noise) else 0.0)
    peaks, _ = find_peaks(y, prominence=prom)
    return peaks


def rolling_shutter_skew(t: np.ndarray, y: np.ndarray, peaks: np.ndarray,
                         max_skew_s_per_px: float) -> tuple[float, float, float]:
    """Time skew between image rows, estimated from the flights themselves.

    Phone cameras read the sensor row by row (rolling shutter): a row lower in the
    image is exposed a little later, up to a whole frame interval at the bottom. A
    falling ball is fast and crosses many rows, so its parabola looks bent unless each
    point gets its own time, t + k * y. We pick k that makes all flights the best
    parabolas. Impacts all happen on the floor row, so this shifts every impact by the
    same amount and leaves flight times (and e) unchanged; it fixes the shape of the
    flights, and with it the impact estimates and the error bars.

    Returns (k in s/px, residual sum of squares at k = 0, at the best k).
    """
    segs = [np.arange(peaks[i] + 1, peaks[i + 1]) for i in range(len(peaks) - 1)]
    segs = [s for s in segs if len(s) >= MIN_ARC_POINTS + 1]
    if not segs:
        return 0.0, float("nan"), float("nan")

    def rss(k):
        total = 0.0
        tt = t + k * y
        for s in segs:
            u = tt[s] - tt[s].mean()
            A = np.column_stack([u**2, u, np.ones_like(u)])
            coef, *_ = np.linalg.lstsq(A, y[s], rcond=None)
            r = y[s] - A @ coef
            total += float(r @ r)
        return total

    grid = np.linspace(-max_skew_s_per_px, max_skew_s_per_px, 81)
    vals = np.array([rss(k) for k in grid])
    i = int(np.argmin(vals))
    k = float(grid[i])
    if 0 < i < len(grid) - 1:  # parabolic refinement between grid points
        a, b, c = vals[i - 1], vals[i], vals[i + 1]
        den = a - 2 * b + c
        if den > 0:
            k = float(grid[i] + 0.5 * (a - c) / den * (grid[1] - grid[0]))
    return k, rss(0.0), rss(k)


@dataclass
class Arc:
    """y = a (t - tc)^2 + b (t - tc) + c for one flight, with the covariance of (a, b, c)."""

    tc: float
    coef: np.ndarray
    cov: np.ndarray
    t_first: float
    t_last: float
    n: int
    rms: float

    def __call__(self, t):
        u = np.asarray(t) - self.tc
        a, b, c = self.coef
        return a * u**2 + b * u + c

    def shifted(self, t_ref: float):
        """Coefficients and covariance in powers of (t - t_ref)."""
        d = t_ref - self.tc
        m = np.array([[1, 0, 0], [2 * d, 1, 0], [d * d, d, 1]], float)
        return m @ self.coef, m @ self.cov @ m.T


def _drop_seen(t, y, first_peak: int) -> bool:
    """Is the motion before the first impact a drop from rest (not a bounce already under way)?

    A drop starts from a ball held still at the top, or at the very first frame. A bounce in
    progress rises first: its highest point is in the middle of the segment, passed quickly.
    """
    seg = np.arange(0, first_peak)
    if len(seg) < 2:
        return False
    i_top = int(np.argmin(y[seg]))
    if i_top <= 1:
        return True
    drop = float(y[first_peak] - y[seg][i_top])
    still = seg[np.abs(y[seg] - y[seg][i_top]) <= 0.01 * drop + 1.0]
    return bool(t[still[-1]] - t[still[0]] >= 0.15)


def _release_segment(t, y, first_peak: int) -> np.ndarray:
    """Indices of the free fall before the first impact (drops frames still in the hand)."""
    seg = np.arange(0, first_peak)
    if len(seg) < MIN_ARC_POINTS:
        return seg[:0]
    top = float(np.min(y[seg]))
    drop = float(y[first_peak] - top)
    held = np.nonzero(y[seg] <= top + 0.05 * drop)[0]
    start = held[-1] + 1 if len(held) else 0
    return seg[start:]


def fit_arcs(t: np.ndarray, y: np.ndarray,
             peaks: np.ndarray) -> tuple[list[Arc | None], float, list[int]]:
    """A parabola through each flight. Arc i ends at impact i (arc 0 is the drop).

    Frames nearest to an impact are left out of both neighbouring flights (the
    ball may be touching the floor, and motion blur is worst there). The noise
    level is pooled over all flights, because a 5-point flight alone says little
    about its own noise. Returns (arcs, pooled sigma in px, indices of flights
    rejected because they do not look like the others); an unusable arc is None.
    """
    bounds = [_release_segment(t, y, peaks[0])]
    for i in range(len(peaks) - 1):
        bounds.append(np.arange(peaks[i] + 1, peaks[i + 1]))
    bounds.append(np.arange(peaks[-1] + 1, len(t)) if len(peaks) else np.arange(0))
    # After the last impact the ball may roll away or leave the frame: keep only a clean rise+fall.
    last = bounds[-1]
    if len(last):
        rest = last[np.argmin(y[last]):]
        stop = np.nonzero(np.diff(y[rest]) <= 0)[0]  # stops falling = next (unseen) impact
        if len(stop):
            last = last[: np.argmin(y[last]) + stop[0] + 1]
        bounds[-1] = last

    def lsq(idx):
        tc = float(np.mean(t[idx]))
        u = t[idx] - tc
        A = np.column_stack([u**2, u, np.ones_like(u)])
        coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
        return tc, A, coef, y[idx] - A @ coef

    raw = []
    for idx in bounds:
        if len(idx) < MIN_ARC_POINTS:
            raw.append(None)
            continue
        tc, A, coef, r = lsq(idx)
        raw.append((idx, tc, A, coef, r))
    fitted = [item for item in raw if item is not None]
    if not fitted:
        return [None] * len(bounds), float("nan"), []

    # A real flight is one clean parabola with the same curvature as the others (same g, same
    # scale). Several tiny bounces merged into one segment, or a hand in the frame, are not.
    rms = np.array([np.sqrt(np.mean(item[4] ** 2)) for item in fitted])
    curv = np.array([item[3][0] for item in fitted])
    ref_rms = max(float(np.median(rms)), 0.1)
    ref_curv = float(np.median(curv))
    rejected = []
    for i, item in enumerate(raw):
        if item is None:
            continue
        r_rms = np.sqrt(np.mean(item[4] ** 2))
        if r_rms > max(5 * ref_rms, 1.0) or abs(item[3][0] - ref_curv) > 0.3 * abs(ref_curv):
            raw[i] = None
            rejected.append(i)

    rss, dof = 0.0, 0
    for item in raw:
        if item is not None:
            rss += float(item[4] @ item[4])
            dof += len(item[0]) - 3
    if dof == 0:
        return [None] * len(bounds), float("nan"), rejected
    sigma = np.sqrt(rss / dof)

    # One pass of outlier removal (> 4 sigma), then the final fits with the pooled sigma.
    rss, dof, kept = 0.0, 0, []
    for item in raw:
        if item is None:
            kept.append(None)
            continue
        idx, tc, A, coef, r = item
        good = np.abs(r) < 4 * sigma
        if good.sum() < len(idx) and good.sum() >= MIN_ARC_POINTS:
            idx = idx[good]
            tc, A, coef, r = lsq(idx)
        kept.append((idx, tc, A, coef, r))
        rss += float(r @ r)
        dof += len(idx) - 3
    sigma = float(np.sqrt(rss / dof)) if dof else float("nan")
    sigma = max(sigma, 0.05)  # never claim better than 1/20 px

    arcs: list[Arc | None] = []
    for item in kept:
        if item is None:
            arcs.append(None)
            continue
        idx, tc, A, coef, r = item
        cov = sigma**2 * np.linalg.inv(A.T @ A)
        arcs.append(Arc(tc, coef, cov, float(t[idx[0]]), float(t[idx[-1]]), len(idx),
                        float(np.sqrt(np.mean(r**2)))))
    return arcs, sigma, rejected


def intersect(left: Arc, right: Arc, t_guess: float) -> tuple[float, float]:
    """Time where two flights meet (the impact) and its 1-sigma error, by error propagation."""
    p, cp = left.shifted(t_guess)
    q, cq = right.shifted(t_guess)
    d = p - q
    roots = np.roots(d) if abs(d[0]) > 1e-12 else np.array([-d[2] / d[1]])
    roots = roots[np.isreal(roots)].real
    if not len(roots):
        raise ValueError("neighbouring flights do not meet")
    u = float(roots[np.argmin(np.abs(roots))])
    slope = 2 * d[0] * u + d[1]
    j = np.array([u * u, u, 1.0])
    var = float(j @ (cp + cq) @ j) / slope**2
    return t_guess + u, float(np.sqrt(var))


# ---------------------------------------------------------- restitution fit ---
def _flights_const(n, T1, e):
    return T1 * e ** np.arange(n)


def _flights_speed(n, T1, e0, k, t_ref):
    """e changes linearly with impact speed; the speed before an impact is g T_prev / 2."""
    out = np.empty(n)
    out[0] = T1
    for j in range(1, n):
        out[j] = out[j - 1] * (e0 + k * (out[j - 1] - t_ref))
    return out


def _times(flights, t1):
    return t1 + np.concatenate([[0.0], np.cumsum(flights)])


MODELS = {
    "constant": "e is the same at every impact",
    "speed-dependent": "e changes linearly with the impact speed",
}


@dataclass
class RestitutionFit:
    name: str
    params: dict
    cov: list[list[float]]
    e: float              # constant model: e; speed model: e at the reference speed
    sigma_e: float
    slope_per_s: float    # de/dT_prev (0 for the constant model)
    sigma_slope: float
    t_ref: float          # reference flight time of the speed model
    chi2: float
    dof: int
    aic: float
    error_scale: float    # sqrt(chi2/dof) when > 1: impact-time errors were inflated by this

    def e_at(self, T_prev: float) -> float:
        return self.e + self.slope_per_s * (T_prev - self.t_ref)


def _aic(chi2: float, k: int) -> float:
    """Akaike information criterion for a fit with known errors (same rule as the pendulum)."""
    return chi2 + 2 * k


def fit_restitution(t_imp: np.ndarray, s_imp: np.ndarray) -> list[RestitutionFit]:
    """Fit impact times t_0..t_{n-1} with each model; best (lowest AIC) first.

    Fitting the times themselves (each has its own independent error) is better
    than averaging flight-time ratios: neighbouring ratios share an impact and
    are correlated.
    """
    t_imp, s_imp = np.asarray(t_imp, float), np.asarray(s_imp, float)
    n = len(t_imp)
    flights = np.diff(t_imp)
    e0 = float(np.median(flights[1:] / flights[:-1])) if n >= 3 else 0.8
    e0 = min(max(e0, 0.1), 0.99)
    out: list[RestitutionFit] = []

    def finish(name, popt, pcov, k, e, se, slope, sslope, t_ref, f):
        r = (t_imp - f(None, *popt)) / s_imp
        chi2 = float(r @ r)
        dof = n - k
        scale = float(np.sqrt(chi2 / dof)) if dof > 0 and chi2 > dof else 1.0
        out.append(RestitutionFit(name, dict(zip(["t1", "T1", "e", "k"], popt)),
                                  (pcov * scale**2).tolist(), float(e), float(se * scale),
                                  float(slope), float(sslope * scale), float(t_ref), chi2, dof,
                                  _aic(chi2, k), scale))

    if n >= 4:
        def fc(_, t1, T1, e):
            return _times(_flights_const(n - 1, T1, e), t1)

        try:
            popt, pcov = curve_fit(fc, None, t_imp, p0=[t_imp[0], flights[0], e0], sigma=s_imp,
                                   absolute_sigma=True, bounds=([-np.inf, 0, 0.01], [np.inf, np.inf, 0.9999]))
            se = float(np.sqrt(pcov[2, 2]))
            finish("constant", popt, pcov, 3, popt[2], se, 0.0, 0.0, 0.0, fc)
        except (RuntimeError, ValueError):
            pass

    if n >= 5:
        t_ref = float(np.median(flights))

        def fs(_, t1, T1, e, k):
            return _times(_flights_speed(n - 1, T1, e, k, t_ref), t1)

        try:
            popt, pcov = curve_fit(fs, None, t_imp, p0=[t_imp[0], flights[0], e0, 0.0],
                                   sigma=s_imp, absolute_sigma=True,
                                   bounds=([-np.inf, 0, 0.01, -5], [np.inf, np.inf, 0.9999, 5]))
            finish("speed-dependent", popt, pcov, 4, popt[2], np.sqrt(pcov[2, 2]), popt[3],
                   np.sqrt(pcov[3, 3]), t_ref, fs)
        except (RuntimeError, ValueError):
            pass
    return sorted(out, key=lambda m: m.aic)


# ----------------------------------------------------------------- physics ---
def simulate_drop(t: np.ndarray, h0: float, e: float, g: float = G_BUSAN, t_release: float = 0.0,
                  e_slope_per_s: float = 0.0, t_ref: float = 0.4, min_flight_s: float = 0.02):
    """Height (m) of the ball's centre above its resting height, for an ideal drop.

    Before t_release the ball is held at h0. e may depend on the impact speed in
    the same way as the speed-dependent model (through the previous flight time).
    Returns (height, impact_times).
    """
    t = np.asarray(t, float)
    h = np.full_like(t, h0)
    t_hit = t_release + np.sqrt(2 * h0 / g)
    fall = (t >= t_release) & (t < t_hit)
    h[fall] = h0 - 0.5 * g * (t[fall] - t_release) ** 2
    impacts = [t_hit]
    T_prev = 2 * np.sqrt(2 * h0 / g)  # a full flight with the same impact speed
    while True:
        T = T_prev * (e + e_slope_per_s * (T_prev - t_ref))
        if T < min_flight_s or impacts[-1] > t[-1]:
            break
        v = g * T / 2
        m = (t >= impacts[-1]) & (t < impacts[-1] + T)
        u = t[m] - impacts[-1]
        h[m] = v * u - 0.5 * g * u**2
        impacts.append(impacts[-1] + T)
        T_prev = T
    h[t >= impacts[-1]] = 0.0
    return h, np.asarray(impacts)


def first_flight(drop_height_m: float, fit: RestitutionFit, g: float = G_BUSAN) -> float:
    """Duration of the first flight after dropping the ball from rest at this height."""
    T_eq = 2 * np.sqrt(2 * drop_height_m / g)  # flight time with the same impact speed
    return float(T_eq * fit.e_at(T_eq))


def predict_first_flight(drop_height_m: float, fit: RestitutionFit, sigma_drop_m: float = 0.005,
                         g: float = G_BUSAN, sigma_g: float = 0.005) -> tuple[float, float]:
    """First flight time after a drop from a new height, with 1-sigma error.

    Errors from e (and its speed slope), the drop height and g are propagated.
    """
    T = first_flight(drop_height_m, fit, g)
    T_eq = 2 * np.sqrt(2 * drop_height_m / g)
    de = fit.slope_per_s * (T_eq - fit.t_ref)  # e at this speed = e + de
    e_here = fit.e + de
    c = np.asarray(fit.cov)
    if fit.name == "constant":
        var_e = c[2, 2]
    else:
        j = np.array([1.0, T_eq - fit.t_ref])
        var_e = float(j @ c[2:4, 2:4] @ j)
    # dT/dT_eq = e_here + T_eq * slope ; T_eq ~ sqrt(h/g)
    dT_dTeq = e_here + T_eq * fit.slope_per_s
    var_Teq = (T_eq / 2) ** 2 * ((sigma_drop_m / drop_height_m) ** 2 + (sigma_g / g) ** 2)
    var = T_eq**2 * var_e + dT_dTeq**2 * var_Teq
    return T, float(np.sqrt(var))


def predict_impacts(t_imp_seen: np.ndarray, fit: RestitutionFit, n_future: int) -> np.ndarray:
    """Times of the next impacts after the observed ones, from the fitted model."""
    flights = np.diff(t_imp_seen)
    T = flights[-1] if len(flights) else fit.params["T1"]
    t = t_imp_seen[-1]
    out = []
    for _ in range(n_future):
        T = T * fit.e_at(T)
        t = t + T
        out.append(t)
    return np.asarray(out)


# ------------------------------------------------------------------ result ---
@dataclass
class BounceResult:
    n_points: int
    noise_px: float
    impacts_s: list[float]
    sigma_impacts_s: list[float]
    flights_s: list[float]
    ratios: list[float]            # T_{n+1}/T_n: one e per impact, model-free
    sigma_ratios: list[float]
    impact_speeds_m_s: list[float]  # g T_prev / 2, before impacts 2.. (uses g, not a scale)
    best_model: str
    models: list[dict]
    e: float
    sigma_e: float
    e_slope_per_s: float
    sigma_e_slope: float
    curvature_px_s2: float         # a of y = a t^2 + ..., equal to g * (px per m) / 2
    curvature_spread_pct: float
    drop_height_m: float | None
    px_per_m: float | None
    g_from_scale: float | None     # rough cross-check, only with --drop-height
    sigma_g_from_scale: float | None
    predicted_first_flight_s: float | None
    sigma_predicted_first_flight_s: float | None
    bounces_end_after_s: float | None  # constant model: T1 / (1 - e) after the first impact
    first_impact_is_drop: bool = False  # impact 1 is the one right after the release
    first_flight_s: float | None = None  # flight right after the drop impact, if seen
    sigma_first_flight_s: float | None = None
    first_flight_method: str = "none"    # "impacts" or "floor level" (drop impact not timed)
    rolling_shutter_s_per_px: float = 0.0  # time skew between rows, applied to the timestamps
    rolling_shutter_readout_ms: float | None = None  # the same over the whole frame height
    reliable: bool = True
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    arcs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_bounce(track: Track, drop_height_m: float | None = None, sigma_drop_m: float = 0.005,
                   g: float = G_BUSAN, frame_height_px: int | None = None,
                   rolling_shutter: bool = True) -> BounceResult:
    tr = track.finite()
    if len(tr.t) < 20:
        raise ValueError(f"only {len(tr.t)} tracked frames; need at least 20")
    t, y = tr.t, tr.y
    warnings: list[str] = []
    if track.coverage() < 0.8:
        warnings.append(f"ball found on only {track.coverage():.0%} of frames")

    noise = noise_px(y)
    peaks = find_impacts(y, noise)
    if len(peaks) < 2:
        raise ValueError(f"found {len(peaks)} impact(s); need at least 2 bounces in the window "
                         "(is the ball moving? is the floor in frame?)")

    # Rolling shutter: allow up to one frame interval of skew over the frame height.
    skew = 0.0
    readout_ms = None
    if rolling_shutter:
        dt = float(np.median(np.diff(t)))
        height = frame_height_px or 1.5 * float(np.ptp(y))
        k, rss0, rss1 = rolling_shutter_skew(t, y, peaks, dt / height)
        n_pts = sum(max(0, b - a - 1) for a, b in zip(peaks[:-1], peaks[1:]))
        # Keep it only when the flights become clearly better parabolas (an AIC gain > 10).
        if np.isfinite(rss1) and rss1 > 0 and n_pts * np.log(rss0 / rss1) > 10:
            skew = k
            t = t + skew * y
            if frame_height_px:
                readout_ms = 1e3 * skew * frame_height_px
    arcs, sigma, rejected = fit_arcs(t, y, peaks)

    # Impact i sits between arc i and arc i+1. Use the longest run of usable neighbours.
    imp_t, imp_s, used = [], [], []
    for i, p in enumerate(peaks):
        left, right = arcs[i], arcs[i + 1]
        if left is None or right is None:
            if imp_t and right is None:
                break  # later flights are too short to see
            continue
        try:
            ti, si = intersect(left, right, float(t[p]))
        except ValueError:
            break
        if not (t[p - 1] - 0.5 * (t[p] - t[p - 1]) <= ti <= t[min(p + 1, len(t) - 1)] + 0.5 * (t[p] - t[p - 1])):
            warnings.append(f"impact {i + 1}: parabolas meet away from the lowest frame; skipped")
            if imp_t:
                break
            continue
        imp_t.append(ti)
        imp_s.append(si)
        used.append(i)
    imp_t, imp_s = np.asarray(imp_t), np.asarray(imp_s)

    problems: list[str] = []
    flights = np.diff(imp_t)
    if len(imp_t) < 4:
        problems.append(f"only {len(imp_t)} impacts timed; need 4 (3 flights). Film at 60 fps, "
                        "drop from higher, keep the whole bounce in frame")

    ratios, s_ratios = [], []
    for j in range(len(imp_t) - 2):
        a_, b_, c_ = imp_t[j:j + 3]
        sa, sb, sc = imp_s[j:j + 3]
        r = (c_ - b_) / (b_ - a_)
        da, dc = (c_ - b_) / (b_ - a_) ** 2, 1 / (b_ - a_)
        db = -da - dc
        ratios.append(float(r))
        s_ratios.append(float(np.sqrt((da * sa) ** 2 + (db * sb) ** 2 + (dc * sc) ** 2)))

    fits = fit_restitution(imp_t, imp_s) if len(imp_t) >= 4 else []
    if fits:
        best = fits[0]
        e, se, slope, sslope = best.e, best.sigma_e, best.slope_per_s, best.sigma_slope
        if best.error_scale > 2:
            warnings.append(f"impact times scatter {best.error_scale:.1f}x more than their "
                            "estimated errors; errors on e were inflated accordingly")
    else:
        best = None
        e = float(np.mean(ratios)) if ratios else float("nan")
        se = float(np.std(ratios) / np.sqrt(len(ratios))) if len(ratios) > 1 else float("nan")
        slope = sslope = 0.0
    if np.isfinite(e) and not 0 < e < 1:
        problems.append(f"e = {e:.3f} is not between 0 and 1: the track is not a bouncing ball")
    elif np.isfinite(e) and e > 0.97:
        problems.append(f"e = {e:.3f}: almost no energy is lost at the impacts, which looks like a "
                        "periodic motion (a swinging object), not a bouncing ball")
    if used:
        odd = [i for i in rejected if 0 < i <= used[-1] + 1]
        if odd:
            problems.append(f"{len(odd)} flight(s) before or inside the timed bounces do not look "
                            "like free fall (wrong curvature or not a parabola): the tracker may "
                            "have followed something else")
        if 0 in rejected:
            warnings.append("the drop before the first impact was not used (hand in the frame?)")

    # Every flight is the same free fall: equal curvature in the image (same g, same scale).
    good = [a for a in arcs if a is not None and a.n >= MIN_ARC_POINTS]
    curv = np.array([a.coef[0] for a in good])
    s_curv = np.array([np.sqrt(a.cov[0, 0]) for a in good])
    w = 1 / s_curv**2
    a_mean = float(np.sum(w * curv) / np.sum(w))
    spread = float(100 * np.std(curv) / abs(a_mean)) if len(curv) > 1 else 0.0
    if a_mean <= 0:
        problems.append("the flights curve upward in the image: this is not a falling object")
    elif len(curv) >= 3 and spread > 15:
        problems.append(f"flights disagree on gravity (curvature spread {spread:.0f} %): the "
                        "tracker may be following something else for part of the video")

    px_per_m = g_scale = s_g_scale = None
    seg0 = _release_segment(t, y, peaks[0])
    if (drop_height_m and arcs[0] is not None and used and used[0] == 0 and len(seg0)
            and seg0[0] > 2 and _drop_seen(t, y, peaks[0])):
        y_release = float(np.median(y[: seg0[0]]))
        y_impact = float(arcs[0](imp_t[0]))
        px_per_m = (y_impact - y_release) / drop_height_m
        if px_per_m > 0:
            g_scale = 2 * a_mean / px_per_m
            s_g_scale = g_scale * float(np.hypot(sigma_drop_m / drop_height_m, np.std(curv) / a_mean))
        else:
            px_per_m = None

    pred = s_pred = None
    if best is not None and drop_height_m:
        pred, s_pred = predict_first_flight(drop_height_m, best, sigma_drop_m, g)
    # The first flight after the drop is what a drop-height prediction is about. When the drop
    # itself is not usable (release out of frame, hand, blur), its start is where the flight's
    # parabola leaves the floor level, measured at the impacts we did time.
    ff = s_ff = None
    ff_method = "none"
    dropped = _drop_seen(t, y, peaks[0])
    if not dropped:
        warnings.append("the recording starts after the drop: the first timed flight is not the "
                        "first flight after the release")
    if not dropped:
        pass
    elif used and used[0] == 0 and len(flights):
        ff, s_ff, ff_method = float(flights[0]), float(np.hypot(imp_s[0], imp_s[1])), "impacts"
    elif used and used[0] == 1 and arcs[1] is not None:
        floor = np.array([arcs[i + 1](ti) for i, ti in zip(used, imp_t)])
        y_floor = float(np.mean(floor))
        s_floor = float(np.std(floor, ddof=1) / np.sqrt(len(floor))) if len(floor) > 1 else sigma
        arc = arcs[1]
        a_, b_, c_ = arc.shifted(arc.t_first)[0]
        roots = np.roots([a_, b_, c_ - y_floor])
        roots = roots[np.isreal(roots)].real
        if len(roots):
            u = float(roots[np.argmin(np.abs(roots))])
            slope = 2 * a_ * u + b_
            t_start = arc.t_first + u
            ff = float(imp_t[0] - t_start)
            s_start = float(np.hypot(s_floor, sigma) / abs(slope))
            s_ff, ff_method = float(np.hypot(s_start, imp_s[0])), "floor level"

    end_after = None
    if best is not None and best.name == "constant" and len(flights):
        end_after = float(flights[0] / (1 - best.e))

    speeds = [float(g * T / 2) for T in flights]
    models = [{"name": m.name, "description": MODELS[m.name], "e": m.e, "sigma_e": m.sigma_e,
               "e_slope_per_s": m.slope_per_s, "sigma_e_slope": m.sigma_slope,
               "e_is_at_flight_s": m.t_ref if m.name != "constant" else None,
               "chi2": round(m.chi2, 2), "dof": m.dof, "aic": round(m.aic, 2),
               "delta_aic": round(m.aic - fits[0].aic, 2)} for m in fits]
    arc_info = [{"t_first": a.t_first, "t_last": a.t_last, "n": a.n, "curvature_px_s2": float(a.coef[0]),
                 "rms_px": a.rms, "tc": a.tc, "coef": a.coef.tolist()} if a is not None else None
                for a in arcs]
    return BounceResult(
        n_points=int(len(t)), noise_px=float(sigma), impacts_s=imp_t.tolist(),
        sigma_impacts_s=imp_s.tolist(), flights_s=flights.tolist(), ratios=ratios,
        sigma_ratios=s_ratios, impact_speeds_m_s=speeds,
        best_model=best.name if best else "none", models=models, e=float(e), sigma_e=float(se),
        e_slope_per_s=float(slope), sigma_e_slope=float(sslope), curvature_px_s2=a_mean,
        curvature_spread_pct=spread, drop_height_m=drop_height_m, px_per_m=px_per_m,
        g_from_scale=g_scale, sigma_g_from_scale=s_g_scale, predicted_first_flight_s=pred,
        sigma_predicted_first_flight_s=s_pred, bounces_end_after_s=end_after,
        first_impact_is_drop=bool(dropped and used and used[0] == 0),
        first_flight_s=ff, sigma_first_flight_s=s_ff, first_flight_method=ff_method,
        rolling_shutter_s_per_px=float(skew), rolling_shutter_readout_ms=readout_ms,
        reliable=not problems, problems=problems, warnings=warnings, arcs=arc_info)