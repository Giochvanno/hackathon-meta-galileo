"""The lab notebook: experiments, the current physical model, and pre-registered predictions.

Everything the agent knows lives here and is saved to one JSON file, so a run can be
inspected, resumed, and audited. Two rules keep the science honest:

* A prediction is frozen when it is made (model coefficients and their covariance
  are copied into it), and stamped with a time.
* A prediction only counts as *blind* if the experiment that tests it was added to
  the notebook after the prediction was made (by the notebook's own event counter)
  and was not used to build the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .pendulum import PendulumResult, analyze_pendulum, local_g
from .track import Track

FOUR_PI2 = 4 * np.pi**2
# A prediction is tested at the experiment's real length (cuts are never exact), but an
# experiment far from the predicted length tests a different claim.
MAX_LENGTH_MISMATCH = 0.20


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def amplitude_factor_deg(amplitude_deg: float) -> float:
    th = np.radians(amplitude_deg)
    return float(1 + th**2 / 16 + 11 * th**4 / 3072)


# ------------------------------------------------------------ length series ---
@dataclass
class SeriesFit:
    """T0^2 = a * L + b. Then g = 4 pi^2 / a and b / a is the effective-length offset."""

    a: float
    b: float
    cov: list[list[float]]   # 2x2 covariance of (a, b)
    g: float
    sigma_g: float
    offset_m: float
    sigma_offset_m: float
    chi2: float
    dof: int
    fit_offset: bool
    n: int

    def predict_T0(self, length_m: float, sigma_length_m: float) -> tuple[float, float]:
        y = self.a * length_m + self.b
        if y <= 0:
            raise ValueError("model predicts a non-positive T0^2 for this length")
        c = np.asarray(self.cov)
        var_y = (length_m**2 * c[0, 0] + c[1, 1] + 2 * length_m * c[0, 1]
                 + (self.a * sigma_length_m) ** 2)
        T0 = float(np.sqrt(y))
        return T0, float(np.sqrt(var_y) / (2 * T0))


def fit_length_series(L, sL, T0, sT0, fit_offset: bool = True) -> SeriesFit:
    """Weighted straight line through (L, T0^2) with errors in both coordinates.

    Length errors are moved onto the y axis through the slope (effective variance
    method) and the fit is iterated, because the slope is what we are solving for.
    Fitting an intercept absorbs any constant error in where L is measured to
    (the bob's centre of mass is never exactly where the tape measure stops).
    """
    L, sL, T0, sT0 = (np.asarray(v, float) for v in (L, sL, T0, sT0))
    y, sy = T0**2, 2 * T0 * sT0
    a = float(np.sum(y * L) / np.sum(L * L))  # start: line through the origin
    b = 0.0
    for _ in range(10):
        w = 1.0 / (sy**2 + (a * sL) ** 2)
        if fit_offset:
            A = np.column_stack([L, np.ones_like(L)])
            cov = np.linalg.inv(A.T @ (A * w[:, None]))
            a, b = cov @ (A.T @ (w * y))
        else:
            var_a = 1.0 / np.sum(w * L * L)
            a = var_a * np.sum(w * L * y)
            cov = np.array([[var_a, 0.0], [0.0, 0.0]])
            b = 0.0
    a, b = float(a), float(b)
    resid = y - (a * L + b)
    chi2 = float(np.sum(w * resid**2))
    dof = len(L) - (2 if fit_offset else 1)
    g = FOUR_PI2 / a
    sigma_g = g * np.sqrt(cov[0, 0]) / a
    offset = b / a if fit_offset else 0.0
    # d(b/a) = db/a - b da / a^2
    jac = np.array([-b / a**2, 1 / a])
    sig_off = float(np.sqrt(jac @ cov @ jac)) if fit_offset else 0.0
    return SeriesFit(a, b, cov.tolist(), float(g), float(sigma_g), float(offset), sig_off,
                     chi2, int(dof), fit_offset, int(len(L)))


# --------------------------------------------------------------------- the lab ---
class Lab:
    def __init__(self, path: str | Path = "runs/lab.json", lat: float = 35.23):
        self.path = Path(path)
        self.lat = lat
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"experiments": {}, "model": None, "predictions": {}, "requests": [],
                          "seq": 0}
        self.state.setdefault("seq", 0)
        self._cache: dict[str, PendulumResult] = {}

    def _next_seq(self) -> int:
        """Order of events in the notebook (clock time can tie within a second)."""
        self.state["seq"] += 1
        return self.state["seq"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))

    # ---- experiments -----------------------------------------------------
    def add_experiment(self, track_csv: str | Path, length_m: float, length_err_m: float,
                       t_start: float | None = None, t_end: float | None = None,
                       note: str = "", amplitude_deg: float | None = None) -> str:
        exps = self.state["experiments"]
        eid = f"E{len(exps) + 1}"
        exps[eid] = {"id": eid, "track_csv": str(track_csv), "length_m": length_m,
                     "length_err_m": length_err_m, "t_start": t_start, "t_end": t_end,
                     "amplitude_deg": amplitude_deg,
                     "note": note, "added_at": _now(), "seq": self._next_seq()}
        self.save()
        return eid

    def experiment(self, eid: str) -> dict:
        try:
            return self.state["experiments"][eid]
        except KeyError:
            raise KeyError(f"no experiment {eid!r}; known: {list(self.state['experiments'])}")

    def analyze(self, eid: str) -> PendulumResult:
        if eid not in self._cache:
            e = self.experiment(eid)
            track = Track.from_csv(e["track_csv"]).clean().window(e["t_start"], e["t_end"])
            self._cache[eid] = analyze_pendulum(track, e["length_m"], e["length_err_m"],
                                                amplitude_deg=e.get("amplitude_deg"),
                                                g_reference=local_g(self.lat))
        return self._cache[eid]

    # ---- model -----------------------------------------------------------
    def estimate_g(self, ids: list[str], fit_offset: bool | None = None) -> dict:
        used, rejected = [], []
        for eid in ids:
            r = self.analyze(eid)
            (used if r.reliable else rejected).append(eid)
        if not used:
            raise ValueError(f"no reliable experiments among {ids}")
        L = [self.experiment(i)["length_m"] for i in used]
        sL = [self.experiment(i)["length_err_m"] for i in used]
        T0 = [self.analyze(i).T0 for i in used]
        sT0 = [self.analyze(i).sigma_T0 for i in used]
        distinct = len({round(v, 3) for v in L})
        if fit_offset is None:
            fit_offset = distinct >= 3
        if fit_offset and distinct < 3:
            raise ValueError("fitting a length offset needs at least 3 different lengths")
        fit = fit_length_series(L, sL, T0, sT0, fit_offset=fit_offset)
        self.state["model"] = {"a": fit.a, "b": fit.b, "cov": fit.cov, "g": fit.g,
                               "sigma_g": fit.sigma_g, "offset_m": fit.offset_m,
                               "sigma_offset_m": fit.sigma_offset_m, "chi2": fit.chi2,
                               "dof": fit.dof, "fit_offset": fit.fit_offset,
                               "basis": used, "created_at": _now()}
        self.save()
        return {**self.state["model"], "rejected_unreliable": rejected,
                "distinct_lengths": distinct}

    def series_fit(self) -> SeriesFit:
        m = self.state["model"]
        if not m:
            raise ValueError("no model yet: estimate g first")
        return SeriesFit(m["a"], m["b"], m["cov"], m["g"], m["sigma_g"], m["offset_m"],
                         m["sigma_offset_m"], m["chi2"], m["dof"], m["fit_offset"],
                         len(m["basis"]))

    # ---- predictions -----------------------------------------------------
    def predict(self, length_m: float, length_err_m: float = 0.003,
                amplitude_deg: float = 10.0) -> dict:
        fit = self.series_fit()
        T0, sT0 = fit.predict_T0(length_m, length_err_m)
        k = amplitude_factor_deg(amplitude_deg)
        preds = self.state["predictions"]
        pid = f"P{len(preds) + 1}"
        preds[pid] = {"id": pid, "length_m": length_m, "length_err_m": length_err_m,
                      "amplitude_deg": amplitude_deg, "T0": T0, "sigma_T0": sT0,
                      "T_at_amplitude": T0 * k, "sigma_T_at_amplitude": sT0 * k,
                      "model": dict(self.state["model"]), "created_at": _now(),
                      "seq": self._next_seq(), "verified_by": None}
        self.save()
        return preds[pid]

    def verify(self, pid: str, eid: str) -> dict:
        try:
            p = self.state["predictions"][pid]
        except KeyError:
            raise KeyError(f"no prediction {pid!r}; known: {list(self.state['predictions'])}")
        e, r = self.experiment(eid), self.analyze(eid)
        m = p["model"]
        frozen = SeriesFit(m["a"], m["b"], m["cov"], m["g"], m["sigma_g"], m["offset_m"],
                           m["sigma_offset_m"], m["chi2"], m["dof"], m["fit_offset"],
                           len(m["basis"]))
        # Re-evaluate the frozen model at the length actually used (cuts are never exact).
        T_pred, s_pred = frozen.predict_T0(e["length_m"], e["length_err_m"])
        z = (r.T0 - T_pred) / float(np.hypot(s_pred, r.sigma_T0))
        blind = e["seq"] > p["seq"] and eid not in m["basis"]
        mismatch = abs(e["length_m"] - p["length_m"]) / p["length_m"]
        if not r.reliable:
            verdict = "inconclusive (measurement unreliable)"
        elif mismatch > MAX_LENGTH_MISMATCH:
            verdict = (f"not a test of this prediction (length differs by {100 * mismatch:.0f} %, "
                       f"limit {100 * MAX_LENGTH_MISMATCH:.0f} %)")
        elif abs(z) < 2:
            verdict = "confirmed"
        elif abs(z) < 3:
            verdict = "tension"
        else:
            verdict = "refuted"
        out = {"prediction": pid, "experiment": eid, "blind": blind,
               "length_predicted_for_m": p["length_m"], "length_used_m": e["length_m"],
               "length_mismatch_pct": 100 * mismatch,
               "note": "the frozen model is re-evaluated at the length actually used",
               "T0_predicted": T_pred, "sigma_T0_predicted": s_pred,
               "T0_measured": r.T0, "sigma_T0_measured": r.sigma_T0,
               "difference_pct": 100 * (r.T0 - T_pred) / T_pred, "z": float(z),
               "verdict": verdict, "verified_at": _now()}
        p["verified_by"] = out
        self.save()
        return out

    def request(self, length_m: float, instructions: str) -> dict:
        req = {"length_m": length_m, "instructions": instructions, "created_at": _now()}
        self.state["requests"].append(req)
        self.save()
        return req
