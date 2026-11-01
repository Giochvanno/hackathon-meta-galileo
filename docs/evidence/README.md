# Evidence: the raw results behind the README

Everything here was produced by the code in this repository from real phone videos (the
videos themselves and the raw SAM 3.1 outputs are kept out of git because of their size).
Each folder is one run: `track.csv` (the SAM 3.1 mask centroid on every frame, with exact
frame times), `result.json` / `result_bounce.json` (every fitted number), `report.png`,
and, where made, `overlay.png` and `look.png` (what Muse Spark saw).

| Folder | What it is |
|---|---|
| `lab.json` | The lab notebook used by the agent: experiments E1–E4, the current model, the frozen predictions P1–P4 with their verdicts, and the experiment requests. |
| `agent/` | Step-by-step logs of every Muse Spark agent run (tool calls, results, tokens, cost). `run_20261101_143109` verified P2; `run_20261101_161903` verified P3. |
| `VID_20261030_180115/` | **E1**, AA battery on a USB cable (L by eye). Includes the SAM text-prompt probes. |
| `experiment01/` | **E2**, golf ball on sewing thread, L = 0.619 m. Tested P1 (confirmed, blind). |
| `VID_20261101_141249/` | **E3**, L = 0.289 m, ~19° release. Tested P2 (confirmed, blind). |
| `VID_20261101_161000/` | **E4**, L = 0.489 m, white ball in front of a white panel. Tested P3 (tension). |
| `p3_auto/` | E4 again, **fully automatic**: Muse Spark chose the frame and the point (`look.json`, `look.png`), SAM 3.1 tracked; same T0 as the human click. |
| `VID_20261101_142028/` | First real ball drop (~50 cm): e = 0.8638 ± 0.0004, rolling shutter measured; prediction B1 made from it. |
| `VID_20261101_152343/` | Three drops "from 30 cm" (the root folder is drop 1, then `drop2/`, `drop3/`): the blind test of B1. |
| `VID_20261030_165152/` | An early battery run, before the length was tape-measured (not in the notebook). |
| `synthetic/`, `synth_live/` | Synthetic videos with a known g, used to validate the pipeline end to end. |

The accuracy map (81 synthetic videos) is in [`../accuracy_opencv.md`](../accuracy_opencv.md).