"""Exercise sam_client against a local fake of the Meta Model API (real OpenAI SDK, real SSE)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from galileo import sam_client

from .helpers import DOC_LINE

LINES = [DOC_LINE.strip(), DOC_LINE.strip().replace("<0f>", "<1f>")]


class FakeMeta(BaseHTTPRequestHandler):
    requests: list = []
    omit_final_output = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/files"):
            FakeMeta.requests.append(("files", self.headers.get("Content-Type", "")))
            out = {"id": "file-123", "object": "file", "bytes": len(body), "created_at": 0,
                   "filename": "video.mp4", "purpose": "user_data", "status": "uploaded"}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        FakeMeta.requests.append(("responses", json.loads(body)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        full = "".join(line + "\n" for line in LINES)
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": full, "annotations": []}]}
        base = {"id": "resp_1", "object": "response", "created_at": 0, "model": "sam-3.1",
                "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}
        seq = iter(range(1000))
        events = [
            {"type": "response.created", "response": {**base, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**msg, "status": "in_progress", "content": []}},
            {"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
            *[{"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0,
               "content_index": 0, "delta": line + "\n", "logprobs": []} for line in LINES],
            {"type": "response.content_part.done", "item_id": "msg_1", "output_index": 0,
             "content_index": 0, "part": msg["content"][0]},
            {"type": "response.output_item.done", "output_index": 0, "item": msg},
            {"type": "response.completed", "response": {
                **base, "status": "completed",
                "output": [] if FakeMeta.omit_final_output else [msg]}},
        ]
        for ev in events:
            ev["sequence_number"] = next(seq)
            self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()


@pytest.fixture
def fake_api(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeMeta)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("META_API_BASE", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    FakeMeta.requests = []
    FakeMeta.omit_final_output = False
    yield FakeMeta
    server.shutdown()


@pytest.mark.parametrize("omit", [False, True])
def test_segment_video_request_and_stream(fake_api, tmp_path, omit):
    fake_api.omit_final_output = omit
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 1024)
    text = sam_client.segment_video(sam_client.get_client(), video, "white earbud", progress=False)
    assert text.splitlines() == LINES

    kind, files_ct = fake_api.requests[0]
    assert kind == "files" and "multipart/form-data" in files_ct
    kind, body = fake_api.requests[1]
    assert body["model"] == "sam-3.1"
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "white earbud"}
    assert content[1] == {"type": "input_video", "file_id": "file-123"}
    assert body["metadata"] == {"include_confidence": "true", "mask_encoding": "one_bit"}
    assert body["stream"] is True


def test_point_prompt_format():
    text = sam_client.point_prompt((1280, 720), 640, 410)
    assert text == "<0f>1<|point;x=640;y=410;w=1280;h=720|>"
