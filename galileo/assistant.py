"""The lab assistant: Muse Spark looks at the video before anything is measured.

It answers the questions a person otherwise answers by clicking through frames:
what kind of experiment is this, which object should be tracked, on which frame
is it free of the hand, where is the pivot or the floor, and is the setup good
enough (camera square to the motion, object in frame, contrast, blur)?

Muse Spark sees several frames at once and points at things on a 0-1000 grid
(dev.meta.ai/docs/image-understanding, "grounding"). We convert those points to
pixels of the exact frames we sent, and hand them to SAM 3.1 as a point prompt.
So nobody has to type coordinates: Spark finds the object, SAM tracks it.

The answer is structured JSON (text.format json_schema, strict). It is cached
next to the run, so looking costs once per video.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ISSUES = ["camera_not_square_to_motion", "object_leaves_frame", "hand_in_frame",
          "low_contrast", "strong_motion_blur", "camera_moving", "pivot_not_visible",
          "floor_not_visible", "cluttered_background", "other"]

POINT = {"type": "object", "additionalProperties": False,
         "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
         "required": ["x", "y"]}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "experiment": {"type": "string", "enum": ["pendulum", "bouncing ball", "other"]},
        "object_name": {"type": "string"},
        "object_frame": {"type": "integer"},
        "object_point": POINT,
        "release_frame": {"type": "integer"},
        "pivot_point": {"anyOf": [POINT, {"type": "null"}]},
        "floor_y": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"kind": {"type": "string", "enum": ISSUES},
                           "detail": {"type": "string"}},
            "required": ["kind", "detail"]}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "summary": {"type": "string"},
    },
    "required": ["experiment", "object_name", "object_frame", "object_point", "release_frame",
                 "pivot_point", "floor_y", "issues", "confidence", "summary"],
}

PROMPT = """\
You are the lab assistant of a physics measurement instrument. A phone filmed one
experiment; you see {n} frames from it, in time order, labelled F0..F{last} with their times.
Coordinates: each frame on a 0-1000 grid, (0,0) top-left, x to the right, y down.

Answer as JSON:
- experiment: "pendulum" (a bob on a string swinging), "bouncing ball" (a ball dropped on
  a floor), or "other".
- object_name: the moving object to measure (the bob or the ball), in a few words.
- object_frame: the index of the EARLIEST frame where that object is clearly visible and
  already free of the hand (released and moving, or about to move freely).
- object_point: the centre of the object in that frame.
- release_frame: the index of the first frame where the object moves on its own.
- pivot_point: for a pendulum, where the string is attached (in object_frame), else null.
- floor_y: for a bouncing ball, the y of the floor line where the ball lands, else null.
- issues: problems that would make the measurement worse, each with a short detail.
  Look for: the camera not facing the plane of motion squarely, the object leaving the
  frame, a hand or person in the frame during the motion, low contrast with the
  background, strong motion blur, a moving camera, the pivot or floor not visible.
- confidence: how sure you are about object_point.
- summary: one or two sentences for the experimenter.
"""


@dataclass
class Look:
    experiment: str
    object_name: str
    frame_index: int          # index into the video
    point_px: tuple[int, int]
    release_index: int
    t_release: float
    pivot_px: tuple[int, int] | None
    floor_y_px: int | None
    issues: list[dict] = field(default_factory=list)
    confidence: str = "low"
    summary: str = ""
    model: str = ""
    usage: dict = field(default_factory=dict)
    sampled_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sample_frames(n_frames: int, k: int = 8, end_fraction: float = 0.7) -> list[int]:
    """Frames spread over the first part of the video (the release is never at the end)."""
    last = max(1, int(n_frames * end_fraction))
    return sorted({int(round(v)) for v in np.linspace(0, last - 1, k)})


def _jpeg_data_url(frame: np.ndarray, max_side: int = 960) -> str:
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def to_px(p: dict, width: int, height: int) -> tuple[int, int]:
    """0-1000 grid -> pixels of the frame we sent; values slightly outside are clamped."""
    x = min(max(p["x"], 0), 1000) / 1000 * (width - 1)
    y = min(max(p["y"], 0), 1000) / 1000 * (height - 1)
    return int(round(x)), int(round(y))


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def look(video_path: str | Path, t: np.ndarray, frames_idx: list[int] | None = None,
         client=None, model: str | None = None, effort: str | None = None) -> Look:
    from . import video as vid

    if client is None:
        from .sam_client import get_client

        client = get_client()
    model = model or os.environ.get("MODEL_LOOK") or os.environ.get("MODEL_AGENT", "muse-spark-1.3")
    effort = effort or os.environ.get("LOOK_REASONING", "medium")
    idx = frames_idx or sample_frames(len(t))
    frames = [vid.read_frame(video_path, i) for i in idx]
    h, w = frames[0].shape[:2]

    content: list[dict] = [{"type": "input_text",
                            "text": PROMPT.format(n=len(idx), last=len(idx) - 1)}]
    for k, (i, fr) in enumerate(zip(idx, frames)):
        content.append({"type": "input_text", "text": f"F{k} (t = {t[i]:.2f} s)"})
        content.append({"type": "input_image", "image_url": _jpeg_data_url(fr)})
    kwargs = dict(model=model, input=[{"role": "user", "content": content}],
                  reasoning={"effort": effort}, max_output_tokens=8000, store=False)
    fmt = {"format": {"type": "json_schema", "name": "lab_setup", "schema": SCHEMA,
                      "strict": True}}
    try:
        resp = client.responses.create(text=fmt, **kwargs)
    except Exception as e:  # noqa: BLE001 - fall back to plain JSON in the text
        if "400" not in str(e) and "format" not in str(e).lower():
            raise
        content[0]["text"] += "\nReply with the JSON object only."
        resp = client.responses.create(**kwargs)
    data = _parse(resp.output_text or "")

    n = len(idx)
    of = min(max(int(data["object_frame"]), 0), n - 1)
    rf = min(max(int(data["release_frame"]), 0), n - 1)
    pivot = to_px(data["pivot_point"], w, h) if data.get("pivot_point") else None
    floor = None
    if data.get("floor_y") is not None:
        floor = to_px({"x": 0, "y": data["floor_y"]}, w, h)[1]
    u = getattr(resp, "usage", None)
    usage = {"input_tokens": getattr(u, "input_tokens", 0),
             "output_tokens": getattr(u, "output_tokens", 0)} if u else {}
    return Look(experiment=data["experiment"], object_name=data["object_name"],
                frame_index=idx[of], point_px=to_px(data["object_point"], w, h),
                release_index=idx[rf], t_release=float(t[idx[rf]]), pivot_px=pivot,
                floor_y_px=floor, issues=list(data.get("issues", [])),
                confidence=data["confidence"], summary=data.get("summary", ""), model=model,
                usage=usage, sampled_frames=idx)


def annotate(frame: np.ndarray, lk: Look, path: str | Path) -> Path:
    """What the assistant saw: the object point, the pivot or floor, and its notes."""
    out = frame.copy()
    x, y = lk.point_px
    cv2.circle(out, (x, y), 18, (52, 104, 235), 3, cv2.LINE_AA)
    cv2.drawMarker(out, (x, y), (52, 104, 235), cv2.MARKER_CROSS, 14, 2)
    if lk.pivot_px:
        cv2.drawMarker(out, lk.pivot_px, (214, 120, 42), cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.line(out, lk.pivot_px, (x, y), (214, 120, 42), 1, cv2.LINE_AA)
    if lk.floor_y_px is not None:
        cv2.line(out, (0, lk.floor_y_px), (out.shape[1], lk.floor_y_px), (214, 120, 42), 2)
    lines = [f"Muse Spark: {lk.experiment}, track '{lk.object_name}' ({lk.confidence})"]
    lines += [f"- {i['kind'].replace('_', ' ')}: {i['detail']}"[:110] for i in lk.issues[:4]]
    for k, text in enumerate(lines):
        org = (16, 34 + 30 * k)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    path = Path(path)
    cv2.imwrite(str(path), out)
    return path


def look_cached(video_path: Path, t: np.ndarray, out: Path, refresh: bool = False) -> Look:
    cache = out / "look.json"
    if cache.exists() and not refresh:
        d = json.loads(cache.read_text())
        d["point_px"] = tuple(d["point_px"])
        d["pivot_px"] = tuple(d["pivot_px"]) if d.get("pivot_px") else None
        return Look(**d)
    lk = look(video_path, t)
    cache.write_text(json.dumps(lk.to_dict(), indent=2))
    return lk