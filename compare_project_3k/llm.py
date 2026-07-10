"""Fireworks access via the OpenAI-compatible SDK.

Everything comes from the environment the harness injects at evaluation
time: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS. Model tiers
are inferred from whatever model IDs arrive in ALLOWED_MODELS — never
hardcoded — so the agent adapts if the list changes on launch day.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

from openai import OpenAI


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader for local runs; real env vars always win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=25.0,      # per-request rule is < 30s
        max_retries=2,
    )


@lru_cache(maxsize=1)
def _allowed() -> tuple[str, ...]:
    raw = os.environ.get("ALLOWED_MODELS", "")
    models = tuple(m.strip() for m in raw.split(",") if m.strip())
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty")
    return models


# --- Tier inference ----------------------------------------------------------
# strong = biggest general model, code = code-specialised (else strong),
# cheap  = fewest active params (MoE-aware), preferring quantized on ties.

_MOE = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")   # 8x7b -> 56
_ACTIVE = re.compile(r"\ba(\d+)b\b")            # ...-a4b -> 4 active
_DENSE = re.compile(r"(\d+)\s*b\b")             # ...-8b -> 8
_CODE = re.compile(r"code|coder")
_QUANT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf")


def _total(mid: str) -> int:
    mid = mid.lower()
    moe = _MOE.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE.finditer(mid)]
    return max(sizes) if sizes else 100


def _active(mid: str) -> int:
    m = _ACTIVE.search(mid.lower())
    return int(m.group(1)) if m else _total(mid)


def _select_tiers(models: list[str]) -> dict[str, str]:
    """Map cheap/strong/code onto concrete IDs. Pure (no env) so tier choice
    is unit-testable against the real harness list without network access.

    Measured, not assumed: routing the strong tier to the biggest general
    model (minimax-m3 on the real list) scored 16/19; diverting it to
    gemma-4-31b-it collapsed to 6/19 — gemma's thinking returns empty content
    under reasoning_effort=none. So minimax-m3 IS the strong tier here; its
    token cost is controlled by the terse prompts, not by avoiding it."""
    coders = [m for m in models if _CODE.search(m.lower())]
    general = [m for m in models if not _CODE.search(m.lower())] or models
    strong = max(general, key=lambda m: (_total(m), not _QUANT.search(m.lower())))
    cheap = min(models, key=lambda m: (_active(m), not _QUANT.search(m.lower())))
    code = max(coders, key=_total) if coders else strong
    return {"cheap": cheap, "strong": strong, "code": code}


@lru_cache(maxsize=1)
def tiers() -> dict[str, str]:
    return _select_tiers(list(_allowed()))


# PREFERRED_MODEL pins every tier to one allowed model chosen for token
# efficiency. Token counts are per-model (each has its own tokenizer), so the
# same text bills differently; measurement picked the leanest allowed model.
# Guarded: if no allowed model matches, tier inference is used unchanged, so
# this can never select an out-of-list model or break on an unexpected list.
_PREFERRED = os.environ.get("PREFERRED_MODEL", "").strip().lower()


def _preferred_model() -> str | None:
    if not _PREFERRED:
        return None
    for m in _allowed():
        if _PREFERRED in m.lower():
            return m
    return None


def model_for(tier: str) -> str:
    explicit = os.environ.get("MODEL") or os.environ.get(f"MODEL_{tier.upper()}")
    if explicit:
        return explicit
    return _preferred_model() or tiers()[tier]


def describe_tiers() -> str:
    return "  ".join(f"{t}={model_for(t)}" for t in ("cheap", "strong", "code"))


# --- Completions -------------------------------------------------------------

_LOCK = threading.Lock()
_USAGE = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
# Models that rejected reasoning_effort; stop sending it to them.
_NO_EFFORT: set[str] = set()
# Thinking tokens are scored; 'none' suppresses hidden reasoning that would
# otherwise drain the budget and sometimes return blank content.
_EFFORT = os.environ.get("REASONING_EFFORT", "none")

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def usage() -> dict[str, int]:
    with _LOCK:
        return dict(_USAGE)


def _record(u) -> None:
    if not u:
        return
    with _LOCK:
        _USAGE["prompt"] += u.prompt_tokens or 0
        _USAGE["completion"] += u.completion_tokens or 0
        _USAGE["total"] += u.total_tokens or 0
        _USAGE["calls"] += 1


def _chat(model: str, messages: list[dict], max_tokens: int) -> str:
    kwargs = {}
    if _EFFORT and model not in _NO_EFFORT:
        kwargs["reasoning_effort"] = _EFFORT
    try:
        resp = _client().chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=0, **kwargs,
        )
    except Exception as exc:
        if kwargs and "reasoning_effort" in str(exc):
            _NO_EFFORT.add(model)
            resp = _client().chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=0,
            )
        else:
            raise
    _record(getattr(resp, "usage", None))
    return _THINK.sub("", resp.choices[0].message.content or "").strip()


def complete(prompt: str, system: str, max_tokens: int, model: str,
             fallback_model: str | None = None) -> str:
    """One completion. A blank answer or a hard failure retries once on the
    fallback model — a blank answer scores zero, so it's worth the tokens."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    use_fb = fallback_model and fallback_model != model
    try:
        answer = _chat(model, messages, max_tokens)
    except Exception:
        if not use_fb:
            raise
        answer = ""
    if not answer and use_fb:
        answer = _chat(fallback_model, messages, max_tokens)
    return answer
