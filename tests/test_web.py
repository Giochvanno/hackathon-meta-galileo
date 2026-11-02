"""The web app: demo mode, limits, and a full job with the pipeline replaced by a fake."""

import io
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    def make(**env):
        for k in ("MODEL_API_KEY", "META_API_KEY", "DEMO_ONLY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GALILEO_DATA", str(tmp_path / "web"))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from galileo import web

        return TestClient(web.create_app()), web
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
    assert client.get(auto["images"][0]).status_code == 200
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
