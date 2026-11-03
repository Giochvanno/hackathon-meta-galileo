"""Re-measure every real run from its saved SAM 3.1 track and compare with the committed result.

No API key and no network: the tracks SAM 3.1 produced are in docs/evidence, and this runs
the same analysis code on them again. A row passes when the re-measured value agrees with
the committed one to well within its own error bar (0.1 sigma), i.e. the published numbers
come from the code in this repository, not from a notebook nobody can rerun.

    python eval/reproduce.py            # prints a table, exits 1 if anything disagrees
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs" / "evidence"

# (folder, what it is, kind, extra settings as recorded in the lab notebook)
RUNS = [
    ("experiment01", "E2 golf ball, 0.619 m (tests P1)", "pendulum", ["--t-start", "5.5"]),
    ("VID_20261101_141249", "E3 golf ball, 0.289 m (tests P2)", "pendulum", []),
    ("VID_20261101_161000", "E4 white ball, 0.489 m (tests P3)", "pendulum", []),
    ("p3_auto", "E4, fully automatic (Muse Spark chose the point)", "pendulum", []),
    ("p4_auto", "E5 white ball, 0.697 m (tests P4), fully automatic", "pendulum", []),
    ("VID_20261101_142028", "bouncing golf ball, first drop", "bounce", []),
]


UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # Windows consoles default to cp1251/cp1252


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "galileo.cli", *args], cwd=ROOT, check=True,
                   capture_output=True, text=True, encoding="utf-8", env=UTF8)


def check(folder: str, kind: str, extra: list[str], tmp: Path) -> list[tuple[str, float, float, float]]:
    out = tmp / folder.replace("/", "_")
    track = str(EV / folder / "track.csv")
    if kind == "pendulum":
        ref = json.loads((EV / folder / "result.json").read_text(encoding="utf-8"))["result"]
        _run(["analyze", track, "--length", str(ref["length_m"]), *extra, "--out", str(out)])
        new = json.loads((out / "result.json").read_text(encoding="utf-8"))["result"]
        return [("T0, s", ref["T0"], new["T0"], ref["sigma_T0"]),
                ("g, m/s²", ref["g"], new["g"], ref["sigma_g"])]
    ref = json.loads((EV / folder / "result_bounce.json").read_text(encoding="utf-8"))["result"]
    _run(["bounce", track, "--out", str(out)])
    new = json.loads((out / "result_bounce.json").read_text(encoding="utf-8"))["result"]
    return [("e", ref["e"], new["e"], ref["sigma_e"])]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    bad = 0
    print(f"{'run':52} {'quantity':9} {'committed':>12} {'re-measured':>12} {'diff/σ':>7}")
    with tempfile.TemporaryDirectory() as t:
        for folder, what, kind, extra in RUNS:
            for name, ref, new, sigma in check(folder, kind, extra, Path(t)):
                z = abs(new - ref) / sigma
                ok = z < 0.1
                bad += not ok
                print(f"{what:52} {name:9} {ref:12.5f} {new:12.5f} {z:7.3f}  {'ok' if ok else 'DIFFERS'}")
    print("all reproduced" if not bad else f"{bad} value(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
