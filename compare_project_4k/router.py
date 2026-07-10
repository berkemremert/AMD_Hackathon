"""Category → model routing. One config dict, easy to tune.

Model values are SUBSTRINGS matched against the ALLOWED_MODELS env var at
runtime, so exact IDs are never hardcoded. If a substring matches nothing,
or the routed model turns out to be undeployed at runtime, callers fall back
through `fallback_models` (the remaining ALLOWED_MODELS in env order) rather
than dropping the task.

Routing decisions (bakeoff, 40-task dev set, see tests/*_results.json):
- Every allowed model emits hidden reasoning by default, billed in
  completion_tokens. reasoning_effort="none" suppresses it on ALL five
  models (it's the only switch that works across families), so every route
  sets it. A route can opt back in with "thinking": True.
- kimi-k2p7-code won or tied every category on accuracy at the lowest total
  tokens, so the map is kimi-only: fewest models, no cross-model surprises.
- math/logic stay accurate with thinking OFF when the instruction asks for
  brief visible working (5/5 at ~1/4 the thinking-ON tokens).
"""

import os

DEFAULT_MODEL = "kimi-k2p7"

# category → routing config. `instruction` becomes the system prompt; keep
# them terse — output tokens count against the score.
ROUTES = {
    "factual": {
        "model": DEFAULT_MODEL,
        "max_tokens": 64,
        "instruction": "Answer directly, no preamble; explanations max 3 short sentences.",
    },
    "math": {
        "model": DEFAULT_MODEL,
        "max_tokens": 384,
        "instruction": "Show brief working, then give the final answer on the last line as: Answer: <value>",
    },
    "sentiment": {
        "model": DEFAULT_MODEL,
        "max_tokens": 40,
        "instruction": "Sentiment label (Positive/Negative/Neutral) only; one justification sentence only if asked.",
    },
    "summarization": {
        "model": DEFAULT_MODEL,
        "max_tokens": 128,
        "instruction": "Obey stated length/format limits exactly; else 2-3 sentences. No preamble.",
    },
    "ner": {
        "model": DEFAULT_MODEL,
        "max_tokens": 200,
        "instruction": "Output entities in exactly the requested format, nothing else.",
    },
    "code_debug": {
        "model": DEFAULT_MODEL,
        "max_tokens": 160,
        "instruction": "Identify the bug and give the corrected code. Be brief.",
    },
    "logic": {
        "model": DEFAULT_MODEL,
        "max_tokens": 416,
        "instruction": "Show brief working, then give the final answer on the last line as: Answer: <value>",
    },
    "code_gen": {
        "model": DEFAULT_MODEL,
        "max_tokens": 320,
        "instruction": "Write exactly the code requested, nothing extra. Code only, minimal comments, no explanation or demos unless asked.",
    },
    "general": {
        "model": DEFAULT_MODEL,
        "max_tokens": 256,
        "instruction": "Answer concisely.",
    },
}

# Suppresses hidden reasoning; works on every allowed model family.
THINKING_OFF = {"reasoning_effort": "none"}


def _allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def resolve_model(substring: str) -> str:
    """Map a config substring to a full model ID from ALLOWED_MODELS."""
    allowed = _allowed_models()
    if not allowed:
        raise RuntimeError("ALLOWED_MODELS env var is empty or unset")
    for model_id in allowed:
        if substring in model_id:
            return model_id
    # Substring matched nothing (config drift / env change): degrade, don't drop.
    return allowed[0]


def route(category: str) -> dict:
    """Return the full routing config for a category.

    Keys: model (full ID), max_tokens, instruction, extra_params,
    fallback_models (remaining ALLOWED_MODELS, in env order).
    """
    cfg = ROUTES.get(category, ROUTES["general"])
    model = resolve_model(cfg["model"])
    return {
        "model": model,
        "max_tokens": cfg["max_tokens"],
        "instruction": cfg["instruction"],
        "extra_params": {} if cfg.get("thinking") else dict(THINKING_OFF),
        "fallback_models": [m for m in _allowed_models() if m != model],
    }
