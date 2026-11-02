"""Web app: film or upload an experiment, get the measurement back.

    uvicorn galileo.web:app --host 0.0.0.0 --port 7860

The page records with the phone's camera in the browser (or takes an uploaded video),
sends it here, and polls a job: Muse Spark looks, SAM 3.1 tracks, the physics is fitted
(galileo/pipeline.py). Real runs cost money, so they are limited:

  MODEL_API_KEY       without it the site runs in demo mode (precomputed examples only)
  DEMO_ONLY=1         force demo mode
  DAILY_BUDGET_USD    spend limit per UTC day across all visitors (default 3)
  RUNS_PER_IP_HOUR    live runs per visitor per hour (default 3)
  MAX_UPLOAD_MB       default 80; videos are also limited to 45 s
  GALILEO_DATA        where jobs and the spend ledger are kept (default runs/web)

The demo examples are the real results committed in docs/evidence: they cost nothing.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
EVIDENCE = ROOT / "docs" / "evidence"
SERVED = {"look.png", "report.png", "overlay.png", "overlay.mp4", "bounce_report.png",
          "summary.json"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Budget:
    """Spend ledger per UTC day, kept on disk so a restart does not reset it."""

    def __init__(self, path: Path, daily_usd: float):
        self.path, self.daily = path, daily_usd
        self.lock = threading.Lock()

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def spent(self) -> float:
        try:
            return float(json.loads(self.path.read_text()).get(self._today(), 0.0))
        except (FileNotFoundError, ValueError):
            return 0.0

    def charge(self, usd: float) -> None:
        """Reserve `usd` or raise if that would pass today's limit."""
        with self.lock:
            try:
                book = json.loads(self.path.read_text())
            except (FileNotFoundError, ValueError):
                book = {}
            day = self._today()
            if book.get(day, 0.0) + usd > self.daily:
                raise BudgetExceeded(f"today's budget for live runs (${self.daily:.2f}) is used up; "
                                     "the examples below still work, and it resets at 00:00 UTC")
            book[day] = round(book.get(day, 0.0) + usd, 4)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(book))


class BudgetExceeded(RuntimeError):
    pass


class Jobs:
    def __init__(self, data: Path, workers: int = 2):
        self.data = data
        self.pool = ThreadPoolExecutor(workers)
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()

    def update(self, jid: str, **kw) -> None:
        with self.lock:
            self.state[jid].update(kw, updated=time.time())

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.state.get(jid)
            return dict(j) if j else None


def create_app() -> FastAPI:
    data = Path(os.environ.get("GALILEO_DATA", ROOT / "runs" / "web"))
    data.mkdir(parents=True, exist_ok=True)
    budget = Budget(data / "budget.json", _env_float("DAILY_BUDGET_USD", 3.0))
    jobs = Jobs(data)
    per_ip = int(_env_float("RUNS_PER_IP_HOUR", 3))
    max_mb = _env_float("MAX_UPLOAD_MB", 80)
    recent: dict[str, deque] = defaultdict(deque)

    def live_enabled() -> tuple[bool, str]:
        if os.environ.get("DEMO_ONLY", "").strip() in ("1", "true", "yes"):
            return False, "this deployment runs in demo mode"
        if not (os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")):
            return False, "no Meta Model API key is configured here, so only the examples run"
        return True, ""

    app = FastAPI(title="Galileo", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/status")
    def status():
        live, why = live_enabled()
        return {"live": live, "reason": why, "budget_usd": budget.daily,
                "spent_today_usd": round(budget.spent(), 2), "runs_per_ip_hour": per_ip,
                "max_seconds": 45, "max_upload_mb": max_mb}

    @app.get("/api/examples")
    def examples():
        return {"examples": _examples()}

    @app.get("/evidence/{path:path}")
    def evidence(path: str):
        f = (EVIDENCE / path).resolve()
        if EVIDENCE.resolve() not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/measure")
    async def measure(request: Request, video: UploadFile = File(...),
                      experiment: str = Form("auto"), length_cm: str = Form(""),
                      drop_cm: str = Form("")):
        live, why = live_enabled()
        if not live:
            raise HTTPException(503, why)
        ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0]
        q = recent[ip]
        now = time.time()
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= per_ip:
            raise HTTPException(429, f"{per_ip} live runs per hour per visitor; try the examples "
                                     "or come back later")
        if experiment not in ("auto", "pendulum", "bouncing ball"):
            raise HTTPException(400, "experiment must be auto, pendulum or bouncing ball")
        length_m = _cm(length_cm, "length")
        drop_m = _cm(drop_cm, "drop height")

        jid = uuid.uuid4().hex[:12]
        jdir = jobs.data / jid
        jdir.mkdir(parents=True)
        suffix = Path(video.filename or "video.mp4").suffix.lower() or ".mp4"
        if suffix not in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".3gp"):
            raise HTTPException(400, "send a video file (mp4, mov, webm)")
        src = jdir / f"upload{suffix}"
        size = 0
        with src.open("wb") as f:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > max_mb * 1e6:
                    f.close()
                    src.unlink(missing_ok=True)
                    raise HTTPException(413, f"the video is larger than {max_mb:.0f} MB")
                f.write(chunk)
        q.append(now)
        with jobs.lock:
            jobs.state[jid] = {"id": jid, "status": "queued", "step": "queued", "detail": "",
                               "created": now, "updated": now, "frames_received": 0}
        jobs.pool.submit(_run_job, jobs, budget, jid, src, jdir, experiment, length_m, drop_m)
        return {"job": jid}

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, "no such job")
        return JSONResponse(j)

    @app.get("/runs/{jid}/{name}")
    def run_file(jid: str, name: str):
        if name not in SERVED or not jid.isalnum():
            raise HTTPException(404)
        f = jobs.data / jid / name
        if not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    return app


def _cm(text: str, what: str) -> float | None:
    text = (text or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        v = float(text) / 100
    except ValueError:
        raise HTTPException(400, f"{what}: enter a number of centimetres")
    if not 0.03 <= v <= 5:
        raise HTTPException(400, f"{what}: {text} cm is outside 3–500 cm")
    return v


def _run_job(jobs: Jobs, budget: Budget, jid: str, src: Path, jdir: Path, experiment: str,
             length_m: float | None, drop_m: float | None) -> None:
    from . import sam_client
    from .pipeline import STEP_TEXT, measure

    def step(name: str, detail: str = "") -> None:
        jobs.update(jid, status="running", step=name, step_text=STEP_TEXT.get(name, name),
                    detail=detail)

    sam_client.set_progress_hook(lambda n: jobs.update(jid, frames_received=n))
    try:
        summary = measure(src, jdir, experiment=experiment, length_m=length_m,
                          length_err_m=0.003, drop_height_m=drop_m, step=step,
                          budget_check=budget.charge)
        files = {f: f"/runs/{jid}/{f}" for f in summary["files"]}
        jobs.update(jid, status="done", step="done", summary=summary, files=files)
    except (ValueError, BudgetExceeded) as e:
        jobs.update(jid, status="error", error=str(e))
    except Exception as e:  # noqa: BLE001 - report, do not crash the server
        (jdir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        jobs.update(jid, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        sam_client.set_progress_hook(None)
        src.unlink(missing_ok=True)  # keep results, not the visitor's video


def _examples() -> list[dict]:
    """Precomputed real runs from docs/evidence (no API calls)."""
    out = []

    def load(rel: str):
        try:
            return json.loads((EVIDENCE / rel).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def pend(folder: str, title: str, story: str):
        d = load(f"{folder}/result.json")
        if not d:
            return
        r = d["result"]
        imgs = [f"/evidence/{folder}/{n}" for n in ("look.png", "overlay.png", "report.png")
                if (EVIDENCE / folder / n).exists()]
        out.append({"id": folder, "kind": "pendulum", "title": title, "story": story,
                    "numbers": {"T0": [r["T0"], r["sigma_T0"]], "g": [r["g"], r["sigma_g"]],
                                "length_m": r["length_m"], "cycles": r["cycles"],
                                "best_model": r["best_model"]},
                    "images": imgs})

    pend("p3_auto", "Pendulum, fully automatic",
         "Muse Spark chose the frame and the ball, SAM 3.1 tracked it: the same period as with "
         "a human click. White ball in front of a white panel.")
    pend("VID_20261101_141249", "Pendulum, blind test of prediction P2",
         "The agent predicted this period the day before; measured within 0.73 σ.")
    pend("experiment01", "Pendulum, blind test of prediction P1",
         "Golf ball on sewing thread, 0.619 m. Prediction confirmed (0.76 σ).")
    d = load("VID_20261101_142028/result_bounce.json")
    if d:
        r = d["result"]
        out.append({"id": "ball", "kind": "bouncing ball", "title": "Bouncing ball",
                    "story": "Eight impacts timed to about a millisecond on a 30 fps phone video; "
                             "the phone's rolling shutter (24 ms) was measured and removed.",
                    "numbers": {"e": [r["e"], r["sigma_e"]], "impacts": len(r["impacts_s"]),
                                "readout_ms": r.get("rolling_shutter_readout_ms")},
                    "images": ["/evidence/VID_20261101_142028/bounce_report.png"]})
    d = load("VID_20261101_152343/drop3/result_bounce.json")
    if d:
        r = d["result"]
        out.append({"id": "b1", "kind": "bouncing ball",
                    "title": "Bouncing ball, a prediction that failed",
                    "story": "Predicted from the drop above: a 30 cm drop gives a first flight of "
                             "0.4275 ± 0.0036 s. This drop gave 0.4558 s (refuted). The instrument "
                             "found why: the ball fell 34 cm, not 30 (from timing and from pixels).",
                    "numbers": {"text": f"first flight {r['first_flight_s']:.4f} ± "
                                        f"{r['sigma_first_flight_s']:.4f} s<br>fell "
                                        f"{100 * r['drop_height_from_timing_m']:.1f} cm (timing), "
                                        f"{100 * r['drop_height_from_pixels_m']:.1f} cm (pixels)"},
                    "images": ["/evidence/VID_20261101_152343/drop3/bounce_report.png"]})
    return out


app = create_app()
