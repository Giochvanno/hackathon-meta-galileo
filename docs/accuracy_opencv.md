# Accuracy map (synthetic videos, known g)

Error = measured g vs true g. Pull = error / reported σ (honest error bars: spread ≈ 1). The length is exact here, so this is the video part of the error.

| Condition | Value | Runs | Median abs error | Worst abs error | Median σ_g / g | Pull spread | Flagged unreliable |
|---|---|---|---|---|---|---|---|
| motion blur | 0 of frame | 3 | 0.004 % | 0.006 % | 0.027 % | 0.2 | 0/3 |
| motion blur | 0.5 of frame (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| motion blur | 1 of frame | 3 | 0.008 % | 0.009 % | 0.026 % | 0.3 | 0/3 |
| sensor noise | 3 grey levels (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| sensor noise | 10 grey levels | 3 | 0.001 % | 0.011 % | 0.027 % | 0.2 | 0/3 |
| sensor noise | 25 grey levels | 3 | 0.040 % | 0.058 % | 0.064 % | 0.6 | 0/3 |
| bob size | 5 px | 3 | 0.004 % | 0.007 % | 0.026 % | 0.2 | 0/3 |
| bob size | 10 px | 3 | 0.007 % | 0.007 % | 0.026 % | 0.2 | 0/3 |
| bob size | 20 px (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| bob size | 40 px | 3 | 0.004 % | 0.005 % | 0.027 % | 0.2 | 0/3 |
| contrast | 120 grey levels (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| contrast | 40 grey levels | 1 | 3698.378 % | 3698.378 % | 76.319 % | 1.3 | 1/1 + 2 failed |
| contrast | 15 grey levels | 0 | — | — | — | — | 3 failed |
| frame rate | 15 fps | 3 | 0.007 % | 0.014 % | 0.026 % | 0.4 | 0/3 |
| frame rate | 30 fps (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| frame rate | 60 fps | 3 | 0.006 % | 0.008 % | 0.026 % | 0.2 | 0/3 |
| duration | 5 s | 3 | 0.009 % | 0.011 % | 0.029 % | 0.3 | 0/3 |
| duration | 10 s | 3 | 0.003 % | 0.009 % | 0.026 % | 0.2 | 0/3 |
| duration | 15 s (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| duration | 30 s | 3 | 0.012 % | 0.012 % | 0.027 % | 0.4 | 0/3 |
| camera shake | 0 px (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| camera shake | 3 px | 3 | 0.004 % | 0.007 % | 0.026 % | 0.2 | 0/3 |
| camera shake | 10 px | 3 | 0.007 % | 0.024 % | 0.027 % | 0.5 | 0/3 |
| camera shake | 25 px | 3 | 0.025 % | 0.034 % | 0.026 % | 1.0 | 0/3 |
| rolling shutter | 0 ms (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| rolling shutter | 15 ms | 3 | 0.004 % | 0.009 % | 0.026 % | 0.2 | 0/3 |
| rolling shutter | 33 ms | 3 | 0.004 % | 0.008 % | 0.026 % | 0.2 | 0/3 |
| swing angle | 5 ° | 3 | 0.007 % | 0.007 % | 0.043 % | 0.2 | 0/3 |
| swing angle | 12 ° (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| swing angle | 25 ° | 3 | 0.012 % | 0.027 % | 0.052 % | 0.3 | 0/3 |
| swing angle | 40 ° | 3 | 0.154 % | 0.154 % | 0.120 % | 1.3 | 0/3 |
| phone tilt | 0 ° (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| phone tilt | 8 ° | 3 | 0.010 % | 0.011 % | 0.026 % | 0.3 | 0/3 |
| phone tilt | 20 ° | 3 | 0.006 % | 0.011 % | 0.026 % | 0.3 | 0/3 |
| damping | 0.02 1/s (baseline) | 3 | 0.005 % | 0.006 % | 0.026 % | 0.2 | 0/3 |
| damping | 0.1 1/s | 3 | 0.028 % | 0.028 % | 0.028 % | 0.9 | 0/3 |
| damping | 0.3 1/s | 3 | 0.701 % | 0.819 % | 45.092 % | 0.0 | 3/3 |