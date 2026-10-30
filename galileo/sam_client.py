"""SAM 3.1 on Meta Model API: one Responses call carries the prompt and the media.

Request shape (dev.meta.ai/docs/sam/segmenting):
    model="sam-3.1", input=[{role: user, content: [input_text, input_video|input_image]}]
    input_text  : a short noun phrase ("white earbud"), OR box/point prompt tokens
    input_video : {"file_id": ...} after POST /v1/files (purpose="user_data"), or {"video_url": ...}
    metadata    : string values only, e.g. {"include_confidence": "true", "mask_encoding": "one_bit"}

The output is special-token text, one line per frame:
    <Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;x=0;y=0;c=..;data=H,W,!payload|>,...
We always save that raw text to disk: re-analysis is free, and we never pay twice.
Pricing (Oct 2026): video $0.20 / 1000 frames, minimum 50 frames per request; image $2.50 / 1000.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
VIDEO_USD_PER_FRAME = 0.20 / 1000
IMAGE_USD = 2.50 / 1000


def sam_model() -> str:
    return os.environ.get("MODEL_SAM", "sam-3.1")


def get_client():
    from openai import OpenAI

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("Set MODEL_API_KEY in .env (key from dev.meta.ai/platform).")
    base = os.environ.get("META_API_BASE", DEFAULT_BASE_URL)
    return OpenAI(base_url=base, api_key=key, timeout=900, max_retries=2)


def estimate_cost_usd(n_frames: int) -> float:
    return max(n_frames, 50) * VIDEO_USD_PER_FRAME


def upload_video(client, path: str | Path) -> str:
    with open(path, "rb") as f:
        return client.files.create(purpose="user_data", file=f).id


def _run(client, content: list[dict], metadata: dict[str, str], progress: bool) -> str:
    """Stream one SAM request and return the full output text."""
    deltas: list[str] = []
    final_text = ""
    with client.responses.stream(
        model=sam_model(),
        input=[{"type": "message", "role": "user", "content": content}],
        metadata=metadata,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                deltas.append(event.delta)
                if progress and len(deltas) % 50 == 0:
                    print(f"  SAM: {len(deltas)} frames received", file=sys.stderr, flush=True)
            elif event.type in ("response.failed", "error"):
                raise RuntimeError(f"SAM request failed: {event}")
        try:
            final_text = stream.get_final_response().output_text or ""
        except Exception:
            final_text = ""
    if final_text:
        return final_text
    # Fallback: assemble the lane ourselves. Each delta is one frame line.
    if any("\n" in d for d in deltas):
        return "".join(deltas)
    return "\n".join(deltas) + ("\n" if deltas else "")


def segment_video(client, video_path: str | Path, text_prompt: str, *,
                  file_id: str | None = None, include_confidence: bool = True,
                  extra_metadata: dict[str, str] | None = None, progress: bool = True) -> str:
    """Segment and track. `text_prompt` is a noun phrase or object-prompt tokens."""
    if file_id is None:
        file_id = upload_video(client, video_path)
    metadata = {"include_confidence": "true" if include_confidence else "false",
                "mask_encoding": "one_bit"}
    metadata.update(extra_metadata or {})
    content = [{"type": "input_text", "text": text_prompt},
               {"type": "input_video", "file_id": file_id}]
    return _run(client, content, metadata, progress)


def segment_image(client, png_bytes: bytes, text_prompt: str, include_confidence: bool = True) -> str:
    data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode()
    content = [{"type": "input_text", "text": text_prompt},
               {"type": "input_image", "image_url": data_url}]
    metadata = {"include_confidence": "true" if include_confidence else "false",
                "mask_encoding": "one_bit"}
    return _run(client, content, metadata, progress=False)


def point_prompt(size: tuple[int, int], x: int, y: int, frame: int = 0, obj_id: int = 1) -> str:
    """Object-prompt text for one positive click on `frame` (coordinates in video pixels)."""
    from meta_sam_parser import FramePrompt, PromptObject, PromptPoint, build_object_prompt

    return build_object_prompt(size=size, objects=[
        PromptObject(id=obj_id, prompts=[FramePrompt(frame=frame, points=[PromptPoint(x, y)])])])


def box_prompt(size: tuple[int, int], box: tuple[int, int, int, int], frame: int = 0,
               obj_id: int = 1) -> str:
    """Object-prompt text for one box (left, top, right, bottom; right/bottom exclusive)."""
    from meta_sam_parser import FramePrompt, PromptBox, PromptObject, build_object_prompt

    return build_object_prompt(size=size, objects=[
        PromptObject(id=obj_id, prompts=[FramePrompt(frame=frame, box=PromptBox(*box))])])
