"""Shared test fixtures: simulated pendulum motion and fake SAM 3.1 wire output."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# A real SAM 3.1 record from dev.meta.ai/docs/sam/reading-segmentation (27 x 48 lossless mask).
DOC_LINE = (
    "<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>"
    ",1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|>"
    "<|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>\n"
)
MASK_48 = "27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@"
MASK_W, MASK_H = 48, 27


def mask_centroid_offset() -> tuple[float, float]:
    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    payload = MASK_48.split(",", 2)[2]
    m = SegmentationMask(encoding="lossless", payload=payload, width=MASK_W, height=MASK_H)
    r = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(MASK_H, MASK_W)
    rows, cols = np.nonzero(r)
    return float(cols.mean()), float(rows.mean())


def simulate_pendulum(t, g=9.7975, length=0.52, theta0_deg=13.0, gamma=0.025):
    def rhs(_, z):
        return [z[1], -g / length * np.sin(z[0]) - 2 * gamma * z[1]]

    sol = solve_ivp(rhs, (0, t[-1] + 1), [np.radians(theta0_deg), 0.0], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    return sol.y[0]


def fake_sam_text(xs, ys, size=(1280, 720), static_xy=(1080, 650), drop_every=0,
                  conf=0.93) -> str:
    """SAM-style video output: moving object id 3, a still object id 7, integer boxes."""
    ox, oy = mask_centroid_offset()
    w, h = size
    lines = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if drop_every and i % drop_every == 5:
            continue  # frame with no detection -> no line at all, like the real API
        recs = []
        for oid, (cx, cy) in ((3, (x, y)), (7, static_xy)):
            l, t = int(round(cx - ox)), int(round(cy - oy))
            recs.append(f"{oid}<|box;x1={l};y1={t};x2={l + MASK_W - 1};y2={t + MASK_H - 1};"
                        f"w={w};h={h};c={conf}|><|mask;x=0;y=0;c={conf};data={MASK_48}|>")
        lines.append(f"<{i}f>" + ",".join(recs))
    return "\n".join(lines) + "\n"
