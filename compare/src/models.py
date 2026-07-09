"""
Model calls.

remote_solve(task) and local_solve(task) both return the SAME shape:
    (answer_string, usage_dict)   with usage_dict = {"prompt_tokens": int, "completion_tokens": int}

REMOTE (Fireworks):
  - If FIREWORKS_API_KEY and FIREWORKS_BASE_URL are set in the environment
    (the judging harness injects these), we make a REAL Fireworks call.
  - Otherwise we fall back to a MOCK, so you can build and test before your
    credits arrive without changing any code.
  - The model is chosen from ALLOWED_MODELS at runtime (never hardcoded), and
    we PREFER a Gemma model to compete for the $6k "Best Use of Gemma" bonus.

LOCAL:
  - Still a MOCK for now. If you decide to add a real bundled local model (the
    free tier), replace _mock_local_solve with a real call. See notes at bottom.
"""

import os
import random

from tokens import count_tokens

# ---------------------------------------------------------------------------
# REMOTE: real Fireworks call, with a mock fallback for local development
# ---------------------------------------------------------------------------

# Safety ceiling on output length (keeps tokens bounded). Tighten per task type
# later to save tokens — shorter answers = fewer tokens = higher rank.
_MAX_OUTPUT_TOKENS = 1024


def _model_preference_order():
    """All allowed models, best-first: Gemma models first (bonus prize), then the rest."""
    allowed = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    gemma = [m for m in allowed if "gemma" in m.lower()]
    rest = [m for m in allowed if "gemma" not in m.lower()]
    return gemma + rest


def choose_model():
    """Kept for compatibility: the single top-preference model."""
    order = _model_preference_order()
    return order[0] if order else None


def _fireworks_solve(task):
    """REAL remote call through the Fireworks proxy the harness provides.

    Tries models in preference order (Gemma first). If one model fails (e.g.
    404 not deployed / temporarily unavailable), it automatically falls back
    to the next allowed model instead of crashing the run.
    """
    from openai import OpenAI   # imported here so local dev works without the package

    from prompts import build_remote_request

    client = OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],       # provided by the harness
        base_url=os.environ["FIREWORKS_BASE_URL"],     # ALL calls must go through this
    )
    messages, max_tokens, _category = build_remote_request(task)

    last_error = None
    resp = None
    for model in _model_preference_order():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            last_error = exc
            continue
    else:
        raise RuntimeError(f"All allowed models failed. Last error: {last_error}")

    answer = _extract_answer(resp)

    # Safety net: some models (e.g. minimax) sometimes put everything in a hidden
    # reasoning field and return empty content. If we STILL have nothing, retry
    # once with a plain, direct instruction — an empty answer is the worst outcome.
    if not answer:
        try:
            retry = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": task["prompt"]}],
                temperature=0,
                max_tokens=max_tokens,
            )
            answer = _extract_answer(retry)
            resp = retry
        except Exception:
            pass

    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return answer, usage


def _extract_answer(resp):
    """Get the clean final answer from a response.

    Normal models put the answer in message.content. 'Reasoning' models (e.g.
    minimax) think out loud in reasoning_content and may leave content empty.
    In that case we pull the useful final answer out of the reasoning:
      - the LAST fenced code block if there is one (code tasks), else
      - text after the last 'Answer:' marker, else
      - the reasoning text itself as a last resort.
    """
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if content:
        return content

    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    if not reasoning:
        return ""

    import re
    # Prefer the last complete ```...``` code block.
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", reasoning, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Else, text after the last "Answer:".
    if "Answer:" in reasoning:
        return reasoning.rsplit("Answer:", 1)[-1].strip()

    return reasoning


def remote_solve(task):
    """Use the real Fireworks model if credentials are present, else the mock."""
    if os.environ.get("FIREWORKS_API_KEY") and os.environ.get("FIREWORKS_BASE_URL"):
        return _fireworks_solve(task)
    return _mock_remote_solve(task)


# ---------------------------------------------------------------------------
# MOCKS (used only for local development before real models/credits are wired)
# ---------------------------------------------------------------------------

_REMOTE_ACCURACY = 0.95
_LOCAL_ACCURACY_EASY = 0.90
_LOCAL_ACCURACY_HARD = 0.35

_WRONG_GUESSES = [
    "unclear", "not certain", "no idea", "hard to say",
    "maybe not", "cannot tell", "unsure", "unknown to me",
]


def _wrong():
    return random.choice(_WRONG_GUESSES)


def _local_accuracy(task):
    return _LOCAL_ACCURACY_HARD if task.get("difficulty") == "hard" else _LOCAL_ACCURACY_EASY


def _mock_remote_solve(task):
    """Fake remote model — pretends to be strong. Used only without real credentials."""
    prompt = task["prompt"]
    correct = str(task.get("answer", "sample-answer"))
    answer = correct if random.random() < _REMOTE_ACCURACY else _wrong()
    usage = {"prompt_tokens": count_tokens(prompt), "completion_tokens": count_tokens(answer)}
    return answer, usage


def _mock_local_solve(task):
    """Fake local model — decent on easy, weak on hard. Free."""
    prompt = task["prompt"]
    correct = str(task.get("answer", "sample-answer"))
    answer = correct if random.random() < _local_accuracy(task) else _wrong()
    usage = {"prompt_tokens": count_tokens(prompt), "completion_tokens": count_tokens(answer)}
    return answer, usage


def local_solve(task):
    """The free local tier. MOCK for now.

    TO ADD A REAL LOCAL MODEL (the free tier that answers easy tasks for zero
    tokens): bundle a small model in the image and run it here, returning the
    same (answer, usage) shape. Keep it small and fast — remember the 10-minute
    total runtime and 30-seconds-per-request limits.
    """
    return _mock_local_solve(task)
