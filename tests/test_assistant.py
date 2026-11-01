"""The lab assistant against a fake Responses server: request shape, grid -> pixels, fallbacks."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from openai import OpenAI

from galileo.assistant import SCHEMA, annotate, look, refine_point, sample_frames, to_px
from galileo.synth import Scene, render

ANSWER = {"experiment": "pendulum", "object_name": "white ball", "object_frame": 1,
          "object_point": {"x": 500, "y": 750}, "release_frame": 0,
          "pivot_point": {"x": 500, "y": 120}, "floor_y": None,
          "issues": [{"kind": "low_contrast", "detail": "grey ball on a grey wall"}],
          "confidence": "high", "summary": "A pendulum; the ball is easy to track."}


class Fake(BaseHTTPRequestHandler):
    requests: list = []
    reject_format = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if Fake.reject_format and "text" in body:
            data = json.dumps({"error": {"message": "unsupported text.format",
                                         "type": "invalid_request_error"}}).encode()
            self.send_response(400)
        else:
            Fake.requests.append(body)
            text = json.dumps(ANSWER)
            if Fake.reject_format:
                text = "Here you go:\n```json\n" + text + "\n```"
            data = json.dumps({
                "id": "resp_1", "object": "response", "created_at": 0, "model": body["model"],
                "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto",
                "tools": [], "output": [{"type": "message", "id": "m", "role": "assistant",
                                         "status": "completed", "content": [
                                             {"type": "output_text", "text": text,
                                              "annotations": []}]}],
                "usage": {"input_tokens": 5000, "output_tokens": 300, "total_tokens": 5300,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 200}}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Fake.requests, Fake.reject_format = [], False
    yield OpenAI(base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="t", max_retries=0)
    server.shutdown()


@pytest.fixture
def video(tmp_path):
    s = Scene(width=320, height=240, px_per_m=400, bob_radius_px=8, seconds=1.0, decoy=False)
    path = tmp_path / "v.mp4"
    render(s, path)
    return path, np.arange(30) / 30.0


def test_grid_to_pixels_is_clamped():
    assert to_px({"x": 0, "y": 0}, 1280, 720) == (0, 0)
    assert to_px({"x": 1000, "y": 1000}, 1280, 720) == (1279, 719)
    assert to_px({"x": 1003, "y": -5}, 1280, 720) == (1279, 0)  # the model can overshoot


def test_frames_come_from_the_first_part_of_the_video():
    idx = sample_frames(1000, k=10)
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] < 700
    assert sum(i < 150 for i in idx) == 5  # dense where the release usually is


def test_point_on_the_edge_of_the_bob_snaps_to_its_centre(tmp_path):
    s = Scene(width=640, height=480, px_per_m=800, length_m=0.4, bob_radius_px=18, seconds=2.0,
              decoy=False, seed=4)
    truth = render(s, tmp_path / "v.mp4")
    x, y = truth["bob_xy"][15]
    (rx, ry), ok = refine_point(tmp_path / "v.mp4", 15, (int(x), int(y - 20)))  # just above it
    assert ok and abs(rx - x) < 3 and abs(ry - y) < 3


def test_annotation_draws_without_errors(client, video, tmp_path):
    path, t = video
    lk = look(path, t, frames_idx=[0, 10, 20], client=client, model="muse-spark-1.3")
    frame = np.full((240, 320, 3), 128, np.uint8)
    assert annotate(frame, lk, tmp_path / "look.png").exists()


def test_look_sends_frames_and_returns_pixels(client, video):
    path, t = video
    lk = look(path, t, frames_idx=[0, 10, 20], client=client, model="muse-spark-1.3")
    body = Fake.requests[0]
    parts = body["input"][0]["content"]
    assert sum(p["type"] == "input_image" for p in parts) == 3
    assert all(p["image_url"].startswith("data:image/jpeg;base64,")
               for p in parts if p["type"] == "input_image")
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"]
    assert body["store"] is False
    assert lk.frame_index == 10 and lk.model_point_px == to_px({"x": 500, "y": 750}, 320, 240)
    assert lk.pivot_px == to_px({"x": 500, "y": 120}, 320, 240) and lk.floor_y_px is None
    assert lk.issues[0]["kind"] == "low_contrast" and lk.usage["input_tokens"] == 5000


def test_falls_back_to_plain_json_when_structured_output_is_refused(client, video):
    Fake.reject_format = True
    path, t = video
    lk = look(path, t, frames_idx=[0, 10, 20], client=client, model="muse-spark-1.3")
    assert "text" not in Fake.requests[0] and lk.object_name == "white ball"


def test_schema_is_strict_compatible():
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                check(v)
        elif isinstance(node, list):
            for v in node:
                check(v)

    check(SCHEMA)
    assert "anyOf" not in SCHEMA and "oneOf" not in json.dumps(SCHEMA)