"""A measured trajectory of one object: time, position, and per-frame quality signals."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Track:
    """Centroid positions of one object, one entry per video frame.

    Coordinates are image pixels (x right, y down). NaN means the object was
    not found on that frame. `area` and `conf` are NaN when the tracker does
    not provide them.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    conf: np.ndarray
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    def coverage(self) -> float:
        return float(self.valid.mean()) if len(self.t) else 0.0

    def _subset(self, m: np.ndarray) -> "Track":
        return Track(self.t[m], self.x[m], self.y[m], self.area[m], self.conf[m],
                     self.source, dict(self.meta))

    def window(self, t_start: float | None = None, t_end: float | None = None) -> "Track":
        m = np.ones(len(self.t), dtype=bool)
        if t_start is not None:
            m &= self.t >= t_start
        if t_end is not None:
            m &= self.t <= t_end
        return self._subset(m)

    def clean(self, min_conf: float = 0.3, area_tol: float = 0.6) -> "Track":
        """Blank out frames whose segmentation looks wrong.

        A frame is dropped when the model's confidence is below `min_conf`, or
        when the mask area is more than `area_tol` (relative) away from the
        median area: a sudden jump in area usually means the mask grabbed the
        wire, a hand, or a shadow.
        """
        bad = np.zeros(len(self.t), dtype=bool)
        if np.isfinite(self.conf).any():
            bad |= np.isfinite(self.conf) & (self.conf < min_conf)
        if np.isfinite(self.area).any():
            med = np.nanmedian(self.area)
            bad |= np.isfinite(self.area) & (np.abs(self.area - med) > area_tol * med)
        x, y = self.x.copy(), self.y.copy()
        x[bad] = np.nan
        y[bad] = np.nan
        out = Track(self.t, x, y, self.area, self.conf, self.source, dict(self.meta))
        out.meta["dropped_frames"] = int(bad.sum())
        return out

    def finite(self) -> "Track":
        return self._subset(self.valid)

    # ---- persistence -------------------------------------------------------
    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t", "x", "y", "area", "conf"])
            for i in range(len(self.t)):
                w.writerow([i, f"{self.t[i]:.6f}"] + [
                    "" if not np.isfinite(v) else f"{v:.4f}"
                    for v in (self.x[i], self.y[i], self.area[i], self.conf[i])
                ])

    @classmethod
    def from_csv(cls, path: str | Path, source: str = "csv") -> "Track":
        cols: dict[str, list[float]] = {k: [] for k in ("t", "x", "y", "area", "conf")}
        with Path(path).open() as f:
            for row in csv.DictReader(f):
                for k in cols:
                    v = row.get(k, "")
                    cols[k].append(float(v) if v not in ("", None) else np.nan)
        arr = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
        return cls(arr["t"], arr["x"], arr["y"], arr["area"], arr["conf"], source)
