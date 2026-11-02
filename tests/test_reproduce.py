"""Every published real result comes out of this code, from the saved SAM 3.1 tracks."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_published_results_reproduce_from_the_saved_tracks():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # σ in the output; Windows pipes are cp1251
    r = subprocess.run([sys.executable, str(ROOT / "eval" / "reproduce.py")], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "all reproduced" in r.stdout
