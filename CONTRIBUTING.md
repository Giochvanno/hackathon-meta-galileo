# Contributing to Isochron AI

Thanks for helping. Isochron AI is a measuring instrument, so the bar is a bit different from
an ordinary app: a change is welcome when it keeps every number honest and checkable.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                               # no network, no API key needed
python eval/reproduce.py                                # every published result, re-measured
```

A Meta Model API key (`cp .env.example .env`) is needed only for runs that call Muse Spark or
SAM 3.1. Never commit `.env`, a key, or anything from `runs/`; CI rejects secret-like strings.

## The rules that keep the numbers honest

1. **Every published number has evidence.** A new result goes into `docs/evidence/<run>/` with
   its `track.csv`, the result JSON and the report, and `eval/reproduce.py` must reproduce it.
2. **Predictions are never edited after the experiment.** A prediction in the lab notebook is
   frozen when it is registered. Measure the length (or height) of a test experiment, and write it
   down, *before* computing the verdict. Failed predictions stay in the README with the reason.
3. **No silent fallbacks.** If a fit is unreliable, the code says so (`reliable: false` and the
   problem), instead of returning a confident number.
4. **Physics changes come with a test against a known answer**, usually a synthetic video or
   track from `galileo/synth.py` or `scripts/make_synthetic_*.py`.

## Pull requests

* Branch from `main`, keep the change small, and describe what changed and how you checked it
  (the PR template asks for this).
* `pytest -q` and `python eval/reproduce.py` must pass; CI runs both on every push.
* Code style: plain Python 3.10+, type hints on public functions, no new heavy dependencies
  without a reason in the PR.

## Reporting a measurement that looks wrong

Open an issue with the *Measurement looks wrong* template: the command you ran, `result.json`,
the report image and, if you can share it, the video. Problems with the video (camera moving,
low contrast, the pivot hidden) are useful to know about too: they become tests.
