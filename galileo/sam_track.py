"""Turn SAM 3.1 output text into per-object trajectories (mask centroids)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import Track


@dataclass(frozen=True)
class Detection:
    frame: int
    object_id: str
    cx: float          # mask centroid, px
    cy: float
    area: int          # mask pixels
    conf: float        # NaN if not requested
    box: tuple[int, int, int, int]   # left, top, right, bottom (right/bottom exclusive)


def parse_sam_output(text: str, media: str = "video") -> tuple[list[Detection], list[str]]:
    """Parse SAM wire text with Meta's official parser and compute mask centroids.

    The mask raster is cropped to the object's box: raster pixel (r, c) sits at
    image pixel (box.top + r, box.left + c). Verified against the docs example.
    """
    from meta_sam_parser import (CompletedOutcome, decode_mask_to_raster,
                                 image_segmentation_format, video_segmentation_format)

    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    list(parser.push(text))
    result = parser.finish(CompletedOutcome()).result
    warnings = [f"{d.severity}: {getattr(d, 'message', d)}" for d in result.diagnostics]

    dets: list[Detection] = []
    for rec in result.records:
        if rec.kind != "mask":
            continue
        b = rec.bounds
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
        raster = raster.reshape(rec.mask.height, rec.mask.width)
        rows, cols = np.nonzero(raster)
        if len(rows):
            cx, cy, area = b.left + cols.mean(), b.top + rows.mean(), int(len(rows))
        else:  # empty mask: fall back to the box centre
            cx, cy, area = (b.left + b.right - 1) / 2, (b.top + b.bottom - 1) / 2, 0
        frame = rec.frame.frame_index if getattr(rec, "frame", None) is not None else 0
        conf = rec.confidence if rec.confidence is not None else np.nan
        dets.append(Detection(frame, rec.object_id, float(cx), float(cy), area, float(conf),
                              (b.left, b.top, b.right, b.bottom)))
    return dets, warnings


def masks_by_frame(text: str, object_id: str | None = None) -> dict[int, tuple[int, int, np.ndarray]]:
    """Per-frame mask of one object: {frame: (left, top, raster)} with raster as uint8 0/1.

    Used to draw what SAM actually saw. object_id None takes the first object found.
    """
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format

    parser = video_segmentation_format().create_parser()
    list(parser.push(text))
    result = parser.finish(CompletedOutcome()).result
    out: dict[int, tuple[int, int, np.ndarray]] = {}
    for rec in result.records:
        if rec.kind != "mask":
            continue
        if object_id is None:
            object_id = rec.object_id
        if rec.object_id != object_id:
            continue
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), dtype=np.uint8)
        out[rec.frame.frame_index] = (rec.bounds.left, rec.bounds.top,
                                      raster.reshape(rec.mask.height, rec.mask.width))
    return out


def tracks_from_detections(dets: list[Detection], t: np.ndarray) -> dict[str, Track]:
    """One Track per SAM object id, aligned to the video's frame timeline."""
    n = len(t)
    tracks: dict[str, Track] = {}
    for d in dets:
        if d.frame >= n:
            continue
        if d.object_id not in tracks:
            nan = np.full(n, np.nan)
            tracks[d.object_id] = Track(t, nan.copy(), nan.copy(), nan.copy(), nan.copy(),
                                        source="sam-3.1", meta={"object_id": d.object_id})
        tr = tracks[d.object_id]
        tr.x[d.frame], tr.y[d.frame] = d.cx, d.cy
        tr.area[d.frame], tr.conf[d.frame] = d.area, d.conf
    return tracks


class NoMovingObjectError(RuntimeError):
    pass


def choose_moving_object(tracks: dict[str, Track],
                         min_spread_px: float = 8.0) -> tuple[str, dict[str, dict]]:
    """Pick the object that is both well tracked and actually moving.

    A wired earphone has two buds; the one lying still on the table scores ~0.
    Score = (frames seen) x (spread of positions).
    """
    summary: dict[str, dict] = {}
    for oid, tr in tracks.items():
        v = tr.valid
        spread = float(np.hypot(np.nanstd(tr.x[v]), np.nanstd(tr.y[v]))) if v.sum() > 2 else 0.0
        summary[oid] = {"frames": int(v.sum()), "spread_px": round(spread, 1),
                        "median_area_px": float(np.nanmedian(tr.area)) if v.any() else 0.0,
                        "score": float(v.sum() * spread)}
    if not summary:
        raise RuntimeError("SAM found no object. Try another prompt (`probe`) or --point X,Y.")
    best = max(summary, key=lambda k: summary[k]["score"])
    if summary[best]["spread_px"] < min_spread_px:
        found = ", ".join(f"id {k}: {v['frames']} frames, moves {v['spread_px']} px"
                          for k, v in summary.items())
        raise NoMovingObjectError(
            f"SAM found only objects that stay still ({found}). The phrase matched something "
            "else in the scene. Check it with `probe`, or click the bob with --point X,Y.")
    return best, summary