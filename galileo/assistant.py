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
    model_point_px: tuple[int, int] | None = None  # where Spark pointed, before snapping
    point_refined: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def sample_frames(n_frames: int, k: int = 10, end_fraction: float = 0.7) -> list[int]:
    """Frames for the assistant: dense at the start, where the release usually is.

    Half of them cover the first 15 % of the video (about one frame every half second in
    a 30 s recording), the rest are spread up to 70 %.
    """
    early = np.linspace(0, max(1, int(n_frames * 0.15)), k // 2, endpoint=False)
    late = np.linspace(int(n_frames * 0.15), max(1, int(n_frames * end_fraction)) - 1, k - k // 2)
    return sorted({int(round(v)) for v in np.concatenate([early, late])})


def refine_point(video_path: str | Path, frame_index: int, point: tuple[int, int],
                 search_px: int = 100, samples: int = 40) -> tuple[tuple[int, int], bool]:
    """Snap the assistant's point to the centre of the moving object next to it.

    Grounding on a 0-1000 grid is coarse, and the model often points at the edge of a ball
    rather than its centre; a point just outside the object makes SAM segment the wrong
    thing (a shelf edge, which then "does not move"). The object moves and the background
    does not, so the median over the video is the empty scene; the object is the compact
    blob that differs from it near the point. Thin strips (edges shifted by a slightly
    moving camera) are ignored. Returns (point, refined?).
    """
    from . import video as vid

    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    idx = [int(i) for i in np.linspace(0, n - 1, samples)]
    frames = vid.read_frames(video_path, idx + [frame_index])
    if frame_index not in frames:
        return point, False
    bg = np.median(np.stack([frames[i] for i in idx if i in frames]), axis=0).astype(np.int16)
    fr = frames[frame_index].astype(np.int16)
    h, w = fr.shape[:2]
    x, y = point
    x0, x1 = max(0, x - search_px), min(w, x + search_px + 1)
    y0, y1 = max(0, y - search_px), min(h, y + search_px + 1)
    diff = np.abs(fr[y0:y1, x0:x1] - bg[y0:y1, x0:x1]).max(axis=2).astype(np.uint8)
    if diff.max() < 25:
        return point, False
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # The object is the *thickest* moving thing near the point: the centre of the largest
    # circle that fits inside the moving region. Thin strips (an edge shifted by a slightly
    # moving camera, a string, fingers) have small inscribed circles.
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    score = dt - 0.15 * np.hypot(xx - (x - x0), yy - (y - y0))
    j = np.unravel_index(int(np.argmax(score)), score.shape)
    if dt[j] < 4:
        return point, False
    # Centre of that blob: centroid of the moving pixels within its inscribed radius * 1.5.
    r = 1.5 * dt[j]
    near = (mask > 0) & (np.hypot(xx - j[1], yy - j[0]) <= r)
    cy, cx = np.argwhere(near).mean(axis=0)
    new = (int(round(cx + x0)), int(round(cy + y0)))
    # A snap is a small correction: at most two object radii (otherwise it found something
    # else, e.g. the hand that is still releasing the ball; then keep the model's point).
    if np.hypot(new[0] - x, new[1] - y) > max(2 * dt[j], 12):
        return point, False
    return new, True


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
    got = vid.read_frames(video_path, idx)
    frames = [got[i] for i in idx]
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
    point = to_px(data["object_point"], w, h)
    refined = False
    try:
        point, refined = refine_point(video_path, idx[of], point)
    except Exception:  # noqa: BLE001 - refinement is a bonus; keep the model's point
        pass
    u = getattr(resp, "usage", None)
    usage = {"input_tokens": getattr(u, "input_tokens", 0),
             "output_tokens": getattr(u, "output_tokens", 0)} if u else {}
    return Look(experiment=data["experiment"], object_name=data["object_name"],
                frame_index=idx[of], point_px=point, model_point_px=to_px(data["object_point"], w, h),
                point_refined=refined,
                release_index=idx[rf], t_release=float(t[idx[rf]]), pivot_px=pivot,
                floor_y_px=floor, issues=list(data.get("issues", [])),
                confidence=data["confidence"], summary=data.get("summary", ""), model=model,
                usage=usage, sampled_frames=idx)


def annotate(frame: np.ndarray, lk: Look, path: str | Path) -> Path:
    """What the assistant saw: the object point, the pivot or floor, and its notes."""
    out = frame.copy()
    x, y = lk.point_px
    if lk.model_point_px and lk.point_refined:
        cv2.drawMarker(out, tuple(lk.model_point_px), (255, 255, 255), cv2.MARKER_SQUARE, 12, 1)
    cv2.circle(out, (x, y), 18, (52, 104, 235), 3, cv2.LINE_AA)
    cv2.drawMarker(out, (x, y), (52, 104, 235), cv2.MARKER_CROSS, 14, 2)
    if lk.pivot_px:
        cv2.drawMarker(out, lk.pivot_px, (214, 120, 42), cv2.MARKER_TILTED_CROSS, 22, 3)
        cv2.line(out, lk.pivot_px, (x, y), (214, 120, 42), 1, cv2.LINE_AA)
    if lk.floor_y_px is not None:
        cv2.line(out, (0, lk.floor_y_px), (out.shape[1], lk.floor_y_px), (214, 120, 42), 2)
    import textwrap

    width = out.shape[1]
    scale = 0.5 if width < 1000 else 0.65
    chars = int(width / (11 * scale / 0.5))  # rough characters per line at this font size
    lines = textwrap.wrap(f"Muse Spark: {lk.experiment}, track '{lk.object_name}' "
                          f"({lk.confidence} confidence)", chars)
    for i in lk.issues[:4]:
        lines += textwrap.wrap(f"- {i['kind'].replace('_', ' ')}: {i['detail']}", chars)
    step = int(26 * scale / 0.5)
    band = out[: 12 + step * len(lines)].astype(np.float32)
    out[: 12 + step * len(lines)] = (band * 0.35).astype(np.uint8)  # dark band under the text
    for k, text in enumerate(lines):
        cv2.putText(out, text, (12, 8 + step * (k + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    path = Path(path)
    cv2.imwrite(str(path), out)
    return path


def look_cached(video_path: Path, t: np.ndarray, out: Path, refresh: bool = False) -> Look:
    cache = out / "look.json"
    if cache.exists() and not refresh:
        d = json.loads(cache.read_text())
        d["point_px"] = tuple(d["point_px"])
        d["pivot_px"] = tuple(d["pivot_px"]) if d.get("pivot_px") else None
        if d.get("model_point_px"):
            d["model_point_px"] = tuple(d["model_point_px"])
            # Snapping is free and may have improved since the answer was cached: redo it.
            point, ok = refine_point(video_path, d["frame_index"], d["model_point_px"])
            d["point_px"], d["point_refined"] = (point, ok) if ok else (d["model_point_px"], False)
        return Look(**d)
    lk = look(video_path, t)
    cache.write_text(json.dumps(lk.to_dict(), indent=2))
    return lk