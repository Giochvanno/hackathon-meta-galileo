"""The experimenter: Muse Spark runs the scientific loop through tools.

The model never does arithmetic on the data. Every number it reports comes out
of a tool, and every tool is a plain, tested Python function on the lab notebook.
The model's job is the judgement in between: which experiments to trust, which
damping law the data supports, when there is enough evidence to estimate g,
which experiment to ask for next, and whether a prediction survived its test.

Reference values (the textbook g) are deliberately hidden from the model, so its
estimate cannot drift toward the answer it knows.

    python -m galileo.cli agent --goal "Measure g and test it with a prediction"
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .lab import Lab

# USD per 1M tokens (dev.meta.ai/docs/pricing-rate-limits, Oct 2026): input, cached input, output
PRICES = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}

SYSTEM_PROMPT = """\
You are Galileo, an experimental physicist running a pendulum laboratory. A phone films
each experiment; SAM 3.1 tracks the bob frame by frame; your tools turn those tracks
into measurements. You cannot see the videos or the raw data, only tool results.

How you work:
1. Look at the experiments in the notebook. Analyze each one before you use it.
2. Trust only reliable measurements. If a tool reports problems or warnings, say what
   they mean for the result instead of ignoring them.
3. Decide which damping law the data supports, using the AIC comparison. A difference
   below 2 means the data cannot tell the models apart; say so.
4. Estimate g. With three or more different lengths, fit the length offset too: it
   absorbs the error in where the bob's centre of mass really is.
5. Make a falsifiable prediction for a length that has NOT been measured yet, then ask
   the human to record that experiment. A prediction is only registered through the
   tool, before the experiment exists.
6. If a pending prediction can be tested by an experiment added after it, verify it.

Rules:
- Every number you state must come from a tool result in this conversation. Never
  compute physical results yourself and never assume the textbook value of g.
- Report uncertainties with every measured or predicted value.
- Be brief and concrete. Finish with a short lab report in plain text: what was measured,
  which model won, g with its uncertainty, the prediction or the verification verdict,
  and the single most useful next step.
"""


# ------------------------------------------------------------------- tools ---
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def schema(self) -> dict:
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.parameters}


def _r(x: float | None, nd: int = 5) -> float | None:
    return None if x is None else float(round(x, nd))


def make_tools(lab: Lab) -> dict[str, Tool]:
    def list_experiments() -> dict:
        exps = []
        for e in lab.state["experiments"].values():
            exps.append({"id": e["id"], "length_m": e["length_m"],
                         "length_err_m": e["length_err_m"], "note": e["note"],
                         "added_at": e["added_at"]})
        preds = [{"id": p["id"], "length_m": p["length_m"], "created_at": p["created_at"],
                  "verified": p["verified_by"] is not None}
                 for p in lab.state["predictions"].values()]
        return {"experiments": exps, "predictions": preds,
                "current_model": None if not lab.state["model"] else {
                    "g": _r(lab.state["model"]["g"]), "sigma_g": _r(lab.state["model"]["sigma_g"]),
                    "basis": lab.state["model"]["basis"]}}

    def analyze_experiment(experiment_id: str) -> dict:
        r = lab.analyze(experiment_id)
        e = lab.experiment(experiment_id)
        return {
            "experiment": experiment_id, "length_m": e["length_m"],
            "reliable": r.reliable, "problems": r.problems, "warnings": r.warnings,
            "frames_used": r.n_points, "cycles": _r(r.cycles, 1),
            "period_measured_s": _r(r.T_measured), "sigma_period_s": _r(r.sigma_T_measured),
            "period_zero_crossings_s": _r(r.T_crossings),
            "fit_vs_crossings_agreement_sigma": _r(r.consistency_z, 2),
            "swing_amplitude_deg": _r(r.amplitude_deg, 1),
            "amplitude_source": r.amplitude_source,
            "small_angle_period_T0_s": _r(r.T0), "sigma_T0_s": _r(r.sigma_T0),
            "best_damping_model": r.best_model,
            "damping_rate_per_s": _r(r.damping_per_s, 4), "quality_factor": _r(r.quality_factor, 1),
            "g_from_this_experiment": _r(r.g, 4), "sigma_g": _r(r.sigma_g, 4),
        }

    def compare_damping_models(experiment_id: str) -> dict:
        r = lab.analyze(experiment_id)
        d = np.array([m["delta_aic"] for m in r.models])
        w = np.exp(-0.5 * d)
        w = w / w.sum()
        rows = [{"model": m["name"], "delta_aic": _r(m["delta_aic"], 2),
                 "akaike_weight": _r(float(wi), 3), "period_s": _r(m["T"]),
                 "damping_param_per_s": _r(m["damping_per_s"], 4),
                 "residual_rms_px": _r(m["residual_rms_px"], 2)} for m, wi in zip(r.models, w)]
        return {"experiment": experiment_id, "models": rows,
                "laws": {"undamped": "no energy loss", "viscous": "drag force ~ v, exponential decay",
                         "quadratic": "air drag ~ v^2, amplitude ~ 1/(1+kt)",
                         "coulomb": "constant pivot friction, linear decay"},
                "rule": "delta_aic < 2: indistinguishable; 4-7: less support; > 10: rejected"}

    def estimate_g(experiment_ids: list[str], fit_length_offset: bool) -> dict:
        out = lab.estimate_g(experiment_ids, fit_offset=fit_length_offset)
        return {"g": _r(out["g"], 4), "sigma_g": _r(out["sigma_g"], 4),
                "length_offset_m": _r(out["offset_m"], 4),
                "sigma_length_offset_m": _r(out["sigma_offset_m"], 4),
                "chi2": _r(out["chi2"], 2), "dof": out["dof"],
                "fit_length_offset": out["fit_offset"], "used": out["basis"],
                "rejected_unreliable": out["rejected_unreliable"],
                "distinct_lengths": out["distinct_lengths"],
                "method": "weighted fit of T0^2 = (4 pi^2 / g) (L + offset)"}

    def predict_period(length_m: float, amplitude_deg: float) -> dict:
        p = lab.predict(length_m, amplitude_deg=amplitude_deg)
        return {"prediction_id": p["id"], "length_m": p["length_m"],
                "small_angle_period_T0_s": _r(p["T0"]), "sigma_T0_s": _r(p["sigma_T0"]),
                "period_at_amplitude_s": _r(p["T_at_amplitude"]),
                "sigma_period_at_amplitude_s": _r(p["sigma_T_at_amplitude"]),
                "amplitude_deg": amplitude_deg, "registered_at": p["created_at"],
                "note": "frozen now; it can only be confirmed by an experiment added later"}

    def verify_prediction(prediction_id: str, experiment_id: str) -> dict:
        v = lab.verify(prediction_id, experiment_id)
        return {k: (_r(x) if isinstance(x, float) else x) for k, x in v.items()}

    def request_experiment(length_m: float, instructions: str) -> dict:
        lab.request(length_m, instructions)
        return {"status": "sent to the human", "length_m": length_m}

    obj = {"type": "object", "additionalProperties": False}
    eid = {"type": "string", "description": "Experiment id such as E1"}
    tools = [
        Tool("list_experiments", "List the experiments, predictions and current model in the "
             "lab notebook.", {**obj, "properties": {}, "required": []}, list_experiments),
        Tool("analyze_experiment", "Measure one experiment: period by two independent methods, "
             "swing amplitude, damping, small-angle period T0, reliability problems and warnings.",
             {**obj, "properties": {"experiment_id": eid}, "required": ["experiment_id"]},
             analyze_experiment),
        Tool("compare_damping_models", "Compare four damping laws on one experiment with AIC "
             "and Akaike weights.",
             {**obj, "properties": {"experiment_id": eid}, "required": ["experiment_id"]},
             compare_damping_models),
        Tool("estimate_g", "Estimate g from one or more experiments by a weighted fit of "
             "T0^2 against length. Unreliable experiments are left out automatically. "
             "Set fit_length_offset true only with 3+ different lengths. Becomes the model used "
             "for predictions.",
             {**obj, "properties": {
                 "experiment_ids": {"type": "array", "items": eid},
                 "fit_length_offset": {"type": "boolean"}},
              "required": ["experiment_ids", "fit_length_offset"]}, estimate_g),
        Tool("predict_period", "Register a falsifiable prediction of the period for a new length, "
             "using the current model. Returns a prediction id.",
             {**obj, "properties": {
                 "length_m": {"type": "number", "description": "Pendulum length, metres"},
                 "amplitude_deg": {"type": "number", "description": "Expected swing, degrees"}},
              "required": ["length_m", "amplitude_deg"]}, predict_period),
        Tool("verify_prediction", "Test a registered prediction against an experiment. "
             "Reports z-score, verdict and whether the test was blind.",
             {**obj, "properties": {"prediction_id": {"type": "string"}, "experiment_id": eid},
              "required": ["prediction_id", "experiment_id"]}, verify_prediction),
        Tool("request_experiment", "Ask the human to record a new experiment.",
             {**obj, "properties": {
                 "length_m": {"type": "number"},
                 "instructions": {"type": "string",
                                  "description": "Short practical instructions for the recording"}},
              "required": ["length_m", "instructions"]}, request_experiment),
    ]
    return {t.name: t for t in tools}


# -------------------------------------------------------------------- loop ---
@dataclass
class RunLog:
    path: Path
    model: str
    steps: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, kind: str, **data) -> None:
        entry = {"t": round(time.time(), 3), "kind": kind, **data}
        self.steps.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def count(self, usage) -> None:
        if usage is None:
            return
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        det = getattr(usage, "input_tokens_details", None)
        self.cached_tokens += (getattr(det, "cached_tokens", 0) or 0) if det else 0
        odet = getattr(usage, "output_tokens_details", None)
        self.reasoning_tokens += (getattr(odet, "reasoning_tokens", 0) or 0) if odet else 0

    def cost_usd(self) -> float:
        p_in, p_cached, p_out = PRICES.get(self.model, PRICES["muse-spark-1.3"])
        fresh = max(self.input_tokens - self.cached_tokens, 0)
        return (fresh * p_in + self.cached_tokens * p_cached + self.output_tokens * p_out) / 1e6


def _call_tool(tools: dict[str, Tool], name: str, arguments: str) -> tuple[dict, bool]:
    """Run one tool. Errors go back to the model as data, so it can correct itself."""
    if name not in tools:
        return {"error": f"unknown tool {name!r}"}, False
    try:
        args = json.loads(arguments or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        return tools[name].fn(**args), True
    except Exception as e:  # noqa: BLE001 - every failure is reported to the model
        return {"error": f"{type(e).__name__}: {e}"}, False


def _replayable(output, with_reasoning: bool) -> list[dict]:
    """Turn response output items into input items for the next request.

    Only the fields the API needs are kept: status and other read-only fields
    are dropped, so the replay works across servers that validate input strictly.
    """
    items: list[dict] = []
    for o in output:
        if o.type == "function_call":
            items.append({"type": "function_call", "call_id": o.call_id, "name": o.name,
                          "arguments": o.arguments})
        elif o.type == "message":
            text = "".join(getattr(c, "text", "") or "" for c in (o.content or []))
            if text:
                items.append({"role": "assistant", "content": text})
        elif o.type == "reasoning" and with_reasoning and getattr(o, "encrypted_content", None):
            items.append({"type": "reasoning", "id": o.id,
                          "summary": [s.model_dump() for s in (o.summary or [])],
                          "encrypted_content": o.encrypted_content})
    return items


def _create(client, history: list[dict], kwargs: dict, with_reasoning: bool):
    """One model call. Returns None if the server rejects encrypted-reasoning replay."""
    from openai import BadRequestError

    extra = {"include": ["reasoning.encrypted_content"]} if with_reasoning else {}
    try:
        return client.responses.create(input=history, **extra, **kwargs)
    except BadRequestError as e:
        msg = str(e).lower()
        if with_reasoning and any(w in msg for w in ("include", "encrypted", "reasoning")):
            return None
        raise


def run_agent(lab: Lab, goal: str, client=None, model: str | None = None,
              reasoning_effort: str | None = None, max_turns: int = 16,
              log_dir: str | Path = "runs/agent", echo: Callable[[str], None] | None = print) -> dict:
    """Run the tool loop until the model answers without calling a tool."""
    if client is None:
        from .sam_client import get_client

        client = get_client()
    model = model or os.environ.get("MODEL_AGENT", "muse-spark-1.3")
    effort = reasoning_effort or os.environ.get("AGENT_REASONING", "high")
    tools = make_tools(lab)
    schemas = [t.schema() for t in tools.values()]

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl", model)
    log.add("start", goal=goal, model=model, reasoning_effort=effort)
    say = echo or (lambda s: None)

    # Stateless: we keep the whole conversation and send it every turn, with store=False.
    # Chaining with previous_response_id failed in practice ("referenced response not
    # found or expired"), so nothing here depends on the server remembering a response.
    kwargs = {"model": model, "instructions": SYSTEM_PROMPT, "tools": schemas,
              "max_output_tokens": 16000, "store": False}
    if effort and effort != "none":
        kwargs["reasoning"] = {"effort": effort}
    replay_reasoning = True  # carry the model's encrypted reasoning between turns, if supported

    history: list[dict] = [{"role": "user", "content": goal}]
    final_text = ""
    turn = 0
    try:
        for turn in range(1, max_turns + 1):
            resp = _create(client, history, kwargs, replay_reasoning)
            if resp is None:  # the server refused encrypted reasoning: go on without it
                replay_reasoning = False
                log.add("note", text="encrypted reasoning not supported; continuing without it")
                history = [h for h in history if h.get("type") != "reasoning"]
                resp = _create(client, history, kwargs, False)
            log.count(resp.usage)
            calls = [o for o in resp.output if o.type == "function_call"]
            text = resp.output_text or ""
            if text.strip():
                log.add("model_text", turn=turn, text=text)
                say(f"\n[Galileo] {text.strip()}")
            history.extend(_replayable(resp.output, replay_reasoning))
            if not calls:
                if resp.status == "incomplete":
                    log.add("incomplete", details=str(getattr(resp, "incomplete_details", "")))
                final_text = text
                break
            for c in calls:
                result, ok = _call_tool(tools, c.name, c.arguments)
                log.add("tool", turn=turn, name=c.name, arguments=c.arguments, ok=ok,
                        result=result)
                say(f"  -> {c.name}({c.arguments}) {'ok' if ok else 'ERROR'}")
                history.append({"type": "function_call_output", "call_id": c.call_id,
                                "output": json.dumps(result, ensure_ascii=False)})
        else:
            log.add("max_turns_reached", max_turns=max_turns)
    except Exception as e:
        log.add("error", turn=turn, error=f"{type(e).__name__}: {e}",
                input_tokens=log.input_tokens, output_tokens=log.output_tokens,
                cost_usd=round(log.cost_usd(), 6))
        raise RuntimeError(f"agent stopped on turn {turn}: {type(e).__name__}: {e} "
                           f"(the notebook keeps everything done so far; log: {log.path})") from e

    summary = {"final_text": final_text, "turns": turn, "model": model,
               "input_tokens": log.input_tokens, "cached_tokens": log.cached_tokens,
               "output_tokens": log.output_tokens, "reasoning_tokens": log.reasoning_tokens,
               "cost_usd": round(log.cost_usd(), 6), "log": str(log.path)}
    log.add("end", **{k: v for k, v in summary.items() if k != "final_text"})
    say(f"\n[{turn} turns, {log.input_tokens} in / {log.output_tokens} out tokens, "
        f"${summary['cost_usd']:.4f}] log: {log.path}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: python -m galileo.cli agent --goal ...")