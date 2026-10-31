"""Which Meta models can this API key use, and do they call tools?

Costs well under one cent. Run from the project root:

    python scripts/check_models.py
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

TOOL = {
    "type": "function",
    "name": "predict_period",
    "description": "Period of a simple pendulum of a given length, in seconds.",
    "parameters": {
        "type": "object",
        "properties": {"length_m": {"type": "number", "description": "Length in metres"}},
        "required": ["length_m"],
        "additionalProperties": False,
    },
}


def main() -> None:
    load_dotenv()
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        sys.exit("Set MODEL_API_KEY in .env")
    client = OpenAI(base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"),
                    api_key=key, timeout=120, max_retries=0)

    print("1) Models visible to this key:")
    try:
        visible = sorted(m.id for m in client.models.list())
        for mid in visible:
            print("   ", mid)
    except Exception as e:  # noqa: BLE001 - we want to see any failure
        visible = []
        print("    could not list models:", e)

    print("\n2) Tool calling test (asks the model to call predict_period for 0.30 m):")
    runs = [("muse-spark-1.3", None), ("muse-spark-1.3", "low"), ("muse-spark-1.3", "high"),
            ("muse-spark-1.2", None)]
    for model, effort in runs:
        label = f"{model} effort={effort or 'default'}"
        try:
            extra = {"reasoning": {"effort": effort}} if effort else {}
            r = client.responses.create(
                model=model,
                input="A pendulum is 0.30 m long. Use the tool to get its period.",
                tools=[TOOL],
                max_output_tokens=4000,
                **extra,
            )
            calls = [o for o in r.output if o.type == "function_call"]
            if calls:
                c = calls[0]
                print(f"   OK   {label:32} called {c.name}({json.loads(c.arguments)})")
            else:
                print(f"   ??   {label:32} no tool call, status={r.status} "
                      f"{getattr(r, 'incomplete_details', None)}: {(r.output_text or '')[:80]!r}")
            u = r.usage
            if u:
                det = getattr(u, "output_tokens_details", None)
                reasoning = getattr(det, "reasoning_tokens", None) if det else None
                print(f"        tokens in/out: {u.input_tokens}/{u.output_tokens}"
                      f" (reasoning {reasoning})")
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL {label:32} {type(e).__name__}: {str(e)[:200]}")



if __name__ == "__main__":
    main()