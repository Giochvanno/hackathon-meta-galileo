"""The agent loop against a scripted fake of the Meta Responses API (real OpenAI SDK, real HTTP)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

from galileo.agent import _call_tool, make_tools, run_agent
from galileo.lab import Lab

from .test_lab import write_track


def fcall(i, name, args):
    return {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}", "name": name,
            "arguments": json.dumps(args), "status": "completed"}


def message(text):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def reasoning(i):
    return {"type": "reasoning", "id": f"rs_{i}", "summary": [],
            "encrypted_content": f"opaque-{i}"}


SCRIPT = [
    [reasoning(1), fcall(1, "list_experiments", {})],
    [fcall(2, "analyze_experiment", {"experiment_id": "E1"}),
     fcall(3, "compare_damping_models", {"experiment_id": "E1"})],
    [fcall(4, "estimate_g", {"experiment_ids": ["E1"], "fit_length_offset": False})],
    [fcall(5, "predict_period", {"length_m": 0.30, "amplitude_deg": 10}),
     fcall(6, "no_such_tool", {})],
    [fcall(7, "request_experiment", {"length_m": 0.30, "instructions": "Shorten to 30 cm."})],
    [message("Lab report: g measured; prediction P1 registered for 0.30 m.")],
]


class FakeResponses(BaseHTTPRequestHandler):
    requests: list = []
    reject_include = False

    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if FakeResponses.reject_include and "include" in body:
            self._send(400, {"error": {"message": "unsupported include: reasoning.encrypted_content",
                                       "type": "invalid_request_error"}})
            return
        FakeResponses.requests.append(body)
        n = len(FakeResponses.requests)
        out = SCRIPT[n - 1]
        if FakeResponses.reject_include:
            out = [o for o in out if o["type"] != "reasoning"]
        resp = {"id": f"resp_{n}", "object": "response", "created_at": 0, "model": body["model"],
                "status": "completed", "output": out, "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400},
                          "output_tokens": 200, "output_tokens_details": {"reasoning_tokens": 150},
                          "total_tokens": 1200}}
        self._send(200, resp)


@pytest.fixture
def fake_client(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeResponses)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeResponses.requests = []
    FakeResponses.reject_include = False
    yield OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test",
                 max_retries=0)
    server.shutdown()


def test_agent_loop_runs_tools_and_returns_report(tmp_path, fake_client):
    lab = Lab(tmp_path / "lab.json")
    lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002, note="test")
    out = run_agent(lab, "Measure g", client=fake_client, model="muse-spark-1.3",
                    reasoning_effort="high", log_dir=tmp_path / "logs", echo=None)

    reqs = FakeResponses.requests
    assert len(reqs) == 6 and out["turns"] == 6
    assert out["final_text"].startswith("Lab report")
    # Stateless: no server-side chaining, store off, encrypted reasoning requested.
    assert all("previous_response_id" not in r and r["store"] is False for r in reqs)
    assert reqs[0]["include"] == ["reasoning.encrypted_content"]
    assert reqs[0]["input"] == [{"role": "user", "content": "Measure g"}]
    assert reqs[0]["reasoning"] == {"effort": "high"}
    assert {t["name"] for t in reqs[0]["tools"]} >= {"analyze_experiment", "verify_prediction"}
    # Each request carries the whole conversation: reasoning, calls, and every result by call_id.
    hist = reqs[2]["input"]
    assert [h.get("type", h.get("role")) for h in hist] == [
        "user", "reasoning", "function_call", "function_call_output",
        "function_call", "function_call", "function_call_output", "function_call_output"]
    assert hist[1]["encrypted_content"] == "opaque-1"
    assert [h["call_id"] for h in hist if h.get("type") == "function_call_output"] == [
        "call_1", "call_2", "call_3"]
    assert all(set(h) == {"type", "call_id", "name", "arguments"}
               for h in hist if h.get("type") == "function_call")
    analyzed = json.loads(hist[6]["output"])
    assert analyzed["reliable"] is True and analyzed["small_angle_period_T0_s"] > 1.2
    # An unknown tool comes back as an error the model can read, not a crash.
    errs = [json.loads(o["output"]) for o in reqs[4]["input"][-2:]]
    assert "prediction_id" in errs[0] and "error" in errs[1]
    # Side effects landed in the notebook.
    assert "P1" in lab.state["predictions"] and lab.state["requests"][0]["length_m"] == 0.30
    # Tokens and cost: 6 x (600 fresh at $1.25 + 400 cached at $0.15 + 200 out at $4.25) / 1e6.
    assert out["input_tokens"] == 6000 and out["reasoning_tokens"] == 900
    assert out["cost_usd"] == pytest.approx(6 * (600 * 1.25 + 400 * 0.15 + 200 * 4.25) / 1e6)
    log = [json.loads(line) for line in open(out["log"], encoding="utf-8")]
    assert log[0]["kind"] == "start" and log[-1]["kind"] == "end"


def test_falls_back_when_encrypted_reasoning_is_refused(tmp_path, fake_client):
    FakeResponses.reject_include = True
    lab = Lab(tmp_path / "lab.json")
    lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002)
    out = run_agent(lab, "Measure g", client=fake_client, log_dir=tmp_path / "logs", echo=None)
    assert out["final_text"].startswith("Lab report")
    assert all("include" not in r for r in FakeResponses.requests)
    log = [json.loads(line) for line in open(out["log"], encoding="utf-8")]
    assert any(e["kind"] == "note" for e in log)


def test_api_error_keeps_the_log(tmp_path, fake_client, monkeypatch):
    import galileo.agent as agent_mod

    calls = {"n": 0}
    real = agent_mod._create

    def flaky(client, history, kwargs, with_reasoning):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("network down")
        return real(client, history, kwargs, with_reasoning)

    monkeypatch.setattr(agent_mod, "_create", flaky)
    lab = Lab(tmp_path / "lab.json")
    lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002)
    with pytest.raises(RuntimeError, match="turn 3"):
        run_agent(lab, "Measure g", client=fake_client, log_dir=tmp_path / "logs", echo=None)
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    entries = [json.loads(line) for line in open(logs[0], encoding="utf-8")]
    assert entries[-1]["kind"] == "error" and entries[-1]["input_tokens"] == 2000


def test_reference_g_is_hidden_from_the_model(tmp_path):
    lab = Lab(tmp_path / "lab.json")
    lab.add_experiment(write_track(tmp_path / "a.csv", 0.40), 0.40, 0.002)
    tools = make_tools(lab)
    results = [_call_tool(tools, "analyze_experiment", json.dumps({"experiment_id": "E1"}))[0],
               _call_tool(tools, "list_experiments", "{}")[0]]

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    seen = set(keys(results))
    assert not {"g_reference", "g_error_pct", "deviation_sigma"} & seen


def test_bad_arguments_are_reported(tmp_path):
    tools = make_tools(Lab(tmp_path / "lab.json"))
    result, ok = _call_tool(tools, "analyze_experiment", json.dumps({"experiment_id": "E9"}))
    assert not ok and "E9" in result["error"]
    result, ok = _call_tool(tools, "predict_period", "not json")
    assert not ok and "error" in result