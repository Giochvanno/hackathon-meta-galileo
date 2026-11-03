# Evaluation

How we know the numbers are right, in four checks. Everything here can be rerun; only the
SAM 3.1 benchmark costs API credits.

| Check | What it shows | Result | Rerun |
|---|---|---|---|
| **1. Reproduction** | every published real result comes out of this code, from the saved SAM 3.1 tracks | 11 of 11 values identical | `python eval/reproduce.py` (free, no key; also run in CI) |
| **2. Accuracy benchmark** | error and honesty of the error bar on 81 synthetic videos with a known g, one condition made harder at a time | 0.000–0.03 % in normal to hard conditions; SAM 3.1 holds where a model-free tracker loses the object | `python scripts/accuracy_map.py --tracker sam` (~$7.40) or `--tracker opencv` (free) |
| **3. Cross-checks on real video** | two trackers, two period estimators, automatic vs human setup | agree to 0.05 %, 0.2–0.6 σ, and identically | in the README, *Validation* |
| **4. Blind predictions** | the physics predicts experiments that were filmed later | P1, P2 confirmed; P3 and P4 in tension, both 2.2 σ short in the same direction (the tape-measured length, not the clock); B1 refuted (the drop was 34 cm, not 30) | `docs/evidence/lab.json`, agent logs in `docs/evidence/agent/` |

## 1. Reproduction (free)

```bash
python eval/reproduce.py
```

```
run                                                  quantity     committed  re-measured  diff/σ
E2 golf ball, 0.619 m (tests P1)                     T0, s          1.58521      1.58521   0.000  ok
E2 golf ball, 0.619 m (tests P1)                     g, m/s²        9.72467      9.72467   0.000  ok
E3 golf ball, 0.289 m (tests P2)                     T0, s          1.08676      1.08676   0.000  ok
E3 golf ball, 0.289 m (tests P2)                     g, m/s²        9.65020      9.65020   0.000  ok
E4 white ball, 0.489 m (tests P3)                    T0, s          1.39817      1.39817   0.000  ok
E4 white ball, 0.489 m (tests P3)                    g, m/s²        9.87520      9.87520   0.000  ok
E4, fully automatic (Muse Spark chose the point)     T0, s          1.39818      1.39818   0.000  ok
E4, fully automatic (Muse Spark chose the point)     g, m/s²        9.87516      9.87516   0.000  ok
E5 white ball, 0.697 m (tests P4), fully automatic   T0, s          1.66505      1.66505   0.000  ok
E5 white ball, 0.697 m (tests P4), fully automatic   g, m/s²        9.92807      9.92807   0.000  ok
bouncing golf ball, first drop                       e              0.86384      0.86384   0.000  ok
all reproduced
```

Writing this check found one stale file: `docs/evidence/experiment01/result.json` had been saved
by an early run with a hand-set 9° swing angle, while the lab notebook (and the P1 verdict) used
the angle measured from the video, 11.5°. The measured period is the same in both; only the
small-angle correction differs (T0 1.5864 vs 1.5852 s). The file now matches the notebook.

## 2. Accuracy benchmark (synthetic videos with a known g)

`galileo/synth.py` renders pendulum videos with exact nonlinear motion, per-row exposure (shutter
blur, rolling shutter), noise, shake and a chosen contrast. `scripts/accuracy_map.py` changes one
condition at a time from a baseline, 3 seeds each, 81 videos, and runs the normal pipeline.

| Condition | OpenCV (no model) | SAM 3.1 |
|---|---|---|
| Normal conditions | 0.005 % | 0.000–0.010 % |
| Ball 40 grey levels above the background | lost on 2 of 3 | **0.03 %**, 3/3 reliable |
| Ball 15 grey levels above the background | lost on 3 of 3 | **0.025 %**, 3/3 reliable |
| Strong damping (0.3 /s) | flagged unreliable 3/3 | **0.10 %** |
| Heavy sensor noise (σ = 25) | 0.04 % | 0.09 % (error bar slightly optimistic) |
| 40° swing | 0.15 % | 0.15 % |

Robust with both: motion blur over a whole frame, 15–60 fps, 5–30 s, rolling shutter up to
33 ms, a phone rolled by 20°, camera shake up to 25 px. Full tables:
[SAM 3.1](../docs/accuracy_sam.md), [OpenCV](../docs/accuracy_opencv.md); plots:
[SAM 3.1](../docs/accuracy_map_sam.png), [OpenCV](../docs/accuracy_map_opencv.png).

The bouncing ball has its own synthetic checks (README, *Validation*): impacts timed to ≈0.3 ms at
30 fps and e = 0.8001 ± 0.0001 for a true 0.800. Its error bars are checked in CI on every push:
over 60 simulated drops the pulls must scatter with a spread between 0.7 and 1.4
(`tests/test_bounce.py::test_error_bars_are_honest`).

## 3. Cross-checks on real video

* SAM 3.1 and a model-free OpenCV tracker on the same video: T0 = 1.0868 vs 1.0862 s.
* Model fit and zero-crossing regression, on every run: 0.2–0.6 σ apart.
* Fully automatic (Muse Spark picks the object and the frame) vs a human click on the same
  video: T0 = 1.39818 vs 1.39817 s.
* Bouncing ball, SAM 3.1 vs OpenCV: e = 0.8638 vs 0.863.

## 4. Blind predictions

The agent freezes a prediction in the lab notebook before the experiment that tests it is
filmed, and verifies it itself when the experiment is added. The full scorecard, including
the predictions that failed and why, is in the README and on the
[site](https://giochvanno.github.io/hackathon-meta-galileo/#predictions).
