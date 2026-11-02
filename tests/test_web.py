"""The web app: demo mode, limits, and a full job with the pipeline replaced by a fake."""

import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    def make(**env):
        from galileo import web  # importing may read a developer's .env: clear it below

        for k in ("MODEL_API_KEY", "META_API_KEY", "DEMO_ONLY", "DAILY_BUDGET_USD",
                  "RUNS_PER_IP_HOUR", "MAX_UPLOAD_MB"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GALILEO_DATA", str(tmp_path / "web"))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return TestClient(web.create_app(load_env=False)), web
    return make


def test_demo_mode_without_a_key(app_factory):
    client, _ = app_factory()
    page = client.get("/").text
    assert "Isochron AI" in page and "oscillator" in page
    st = client.get("/api/status").json()
    assert st["live"] is False and "key" in st["reason"]
    r = client.post("/api/measure", files={"video": ("v.mp4", b"x", "video/mp4")})
    assert r.status_code == 503


def test_examples_are_real_results(app_factory):
    client, _ = app_factory()
    ex = client.get("/api/examples").json()["examples"]
    kinds = {e["kind"] for e in ex}
    assert {"pendulum", "bouncing ball"} <= kinds
    auto = next(e for e in ex if e["id"] == "p3_auto")
    assert abs(auto["numbers"]["T0"][0] - 1.3982) < 1e-3
    for e in ex:  # every picture exists and says what it shows
        assert e["source"].startswith("https://github.com/") and e["images"]
        for im in e["images"]:
            assert client.get(im["src"]).status_code == 200 and len(im["caption"]) > 40
    assert auto["images"][-1]["src"].endswith(".mp4") and client.get(auto["images"][-1]["poster"]).status_code == 200
    ball = next(e for e in ex if e["id"] == "ball")
    assert [im["label"] for im in ball["images"]] == ["the fit", "the video, frame by frame"]
    assert client.get("/evidence/../pyproject.toml").status_code == 404


def test_rate_limit_and_bad_input(app_factory, monkeypatch):
    client, web = app_factory(MODEL_API_KEY="test", RUNS_PER_IP_HOUR="1")
    monkeypatch.setattr(web, "_run_job", lambda *a, **k: None)  # never the real pipeline
    bad = client.post("/api/measure", data={"length_cm": "abc"},
                      files={"video": ("v.mp4", b"x", "video/mp4")})
    assert bad.status_code == 400
    ok = client.post("/api/measure", files={"video": ("v.mp4", b"x", "video/mp4")})
    assert ok.status_code == 200
    again = client.post("/api/measure", files={"video": ("v.mp4", b"x", "video/mp4")})
    assert again.status_code == 429


def test_budget_refuses_when_used_up(tmp_path):
    from galileo.web import Budget, BudgetExceeded

    b = Budget(tmp_path / "b.json", daily_usd=0.30)
    b.charge(0.25)
    with pytest.raises(BudgetExceeded):
        b.charge(0.10)
    assert Budget(tmp_path / "b.json", 0.30).spent() == pytest.approx(0.25)  # survives a restart


def test_a_job_runs_to_the_end(app_factory, monkeypatch):
    client, web = app_factory(MODEL_API_KEY="test")
    import galileo.pipeline as pipeline

    def fake_measure(src, out, step=None, budget_check=None, **kw):
        step("look", "about $0.02")
        budget_check(0.03)
        (out / "report.png").write_bytes(b"\x89PNG fake")
        step("done")
        return {"experiment": "pendulum", "files": ["report.png"],
                "look": {"object_name": "ball"}, "result": {"g": 9.8}}

    monkeypatch.setattr(pipeline, "measure", fake_measure)
    jid = client.post("/api/measure", data={"length_cm": "48.9"},
                      files={"video": ("v.webm", io.BytesIO(b"x" * 10), "video/webm")}).json()["job"]
    for _ in range(50):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert client.get(j["files"]["report.png"]).content.startswith(b"\x89PNG")
    assert client.get(f"/runs/{jid}/upload.webm").status_code == 404  # never served
    assert client.get("/api/status").json()["spent_today_usd"] == pytest.approx(0.03)


def test_header_trace_and_scorecard_come_from_the_evidence(app_factory):
    client, _ = app_factory()
    h = client.get("/api/hero").json()
    assert len(h["t"]) == len(h["s"]) > 200 and abs(h["T0"] - 1.3982) < 1e-3
    rows = client.get("/api/scorecard").json()["predictions"]
    verdicts = {(r["id"], r["verdict"]) for r in rows}
    assert ("P1", "confirmed") in verdicts and ("P3", "tension") in verdicts
    assert ("B1", "refuted") in verdicts  # failed predictions are shown too


def test_replay_logo_and_favicon(app_factory):
    client, _ = app_factory()
    r = client.get("/api/replay").json()
    assert r["sam"]["frames"] > 900 and r["look"]["model"].startswith("muse-spark")
    assert abs(r["fit"]["T0"] - 1.3982) < 1e-3 and r["verdict"]["verdict"] == "tension"
    assert client.get(r["look"]["image"]).status_code == 200
    for path in ("/static/logo.svg", "/favicon.ico"):
        res = client.get(path)
        assert res.status_code == 200 and "<svg" in res.text


def test_limits_cannot_be_dodged_and_are_checked_before_upload(app_factory, monkeypatch):
    client, web = app_factory(MODEL_API_KEY="test", RUNS_PER_IP_HOUR="2", MAX_UPLOAD_MB="1")
    monkeypatch.setattr(web, "_run_job", lambda *a, **k: None)
    data = Path(web.os.environ["GALILEO_DATA"])

    def post(fake: str, body: bytes = b"x", name: str = "v.mp4"):
        # a visitor can write anything on the left; the hosting proxy appends the real address
        return client.post("/api/measure", files={"video": (name, body, "video/mp4")},
                           headers={"X-Forwarded-For": f"{fake}, 1.1.1.1, 10.0.0.2"})

    codes = [post(f"8.8.8.{i}").status_code for i in range(4)]
    assert codes == [200, 200, 429, 429]  # faking the address does not give new runs

    before = set(data.iterdir())
    assert post("x", name="evil.exe", body=b"x").status_code in (400, 429)
    big = client.post("/api/measure", files={"video": ("v.mp4", b"x" * 3_000_000, "video/mp4")},
                      headers={"X-Forwarded-For": "9.9.9.9"})
    assert big.status_code == 413
    assert set(data.iterdir()) == before  # nothing left behind

    monkeypatch.setattr(web.Jobs, "waiting", lambda self: web.MAX_WAITING_JOBS)
    busy = client.post("/api/measure", files={"video": ("v.mp4", b"x", "video/mp4")},
                       headers={"X-Forwarded-For": "9.9.9.10"})
    assert busy.status_code == 503 and "queue" in busy.json()["detail"]

    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/openapi.json").status_code == 404
