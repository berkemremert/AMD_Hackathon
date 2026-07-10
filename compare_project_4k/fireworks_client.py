"""Thin OpenAI-compatible chat client for Fireworks.

- Reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL from env at call time
  (with a .env fallback for local dev — see load_dotenv_fallback).
- temperature=0 always.
- Retries transient failures (429, 5xx, connection errors) with backoff.
- Appends model + token usage for every successful call to tokens.log.
"""

import json
import os
import sys
import threading
import time

import requests

TOKENS_LOG = os.environ.get("TOKENS_LOG", "tokens.log")
_log_lock = threading.Lock()

MAX_RETRIES = 4
BACKOFF_BASE = 2.0  # 1s, 2s, 4s, 8s
REQUEST_TIMEOUT = 120

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ModelUnavailableError(RuntimeError):
    """The model ID itself is missing/undeployed — retrying won't help,
    but falling back to a different allowed model might."""


class BadRequestError(RuntimeError):
    """HTTP 400 — the request body itself was rejected."""


# Set to True the first time the endpoint 400s a request that carried
# extra_params but accepts it without them (e.g. a harness proxy that rejects
# reasoning_effort). From then on extra_params are dropped for the whole run
# instead of paying a doubled call every time. Plain bool write: atomic enough
# under threads; worst case a couple of extra degrade cycles race in parallel.
_disable_extra_params = False


def load_dotenv_fallback(path=".env"):
    """Local-dev only: populate os.environ from .env for keys not already set.

    Never required in the judged environment — real env vars always win.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _log_usage(model, usage):
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    with _log_lock:
        with open(TOKENS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")


def chat(model, prompt, instruction=None, max_tokens=256, extra_params=None):
    """Single chat completion. Returns the answer text.

    Raises RuntimeError after exhausting retries; callers handle per-task.
    ModelUnavailableError (subclass) signals the model itself is missing, so
    callers can fall back to another allowed model.
    """
    return chat_with_usage(model, prompt, instruction, max_tokens, extra_params)[0]


def chat_with_usage(model, prompt, instruction=None, max_tokens=256, extra_params=None):
    """Like chat(), but returns (text, usage_dict) for token accounting.

    If the endpoint 400s a request carrying extra_params (a proxy that
    rejects e.g. reasoning_effort), the identical request is retried once
    without them, and extra_params are disabled for the rest of the run.
    """
    global _disable_extra_params
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
    if not api_key or not base_url:
        raise RuntimeError("FIREWORKS_API_KEY / FIREWORKS_BASE_URL not set")

    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": prompt})

    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{base_url}/chat/completions"

    use_extras = bool(extra_params) and not _disable_extra_params
    if not use_extras:
        return _request(model, url, headers, base_payload)

    payload = dict(base_payload)
    payload.update(extra_params)
    try:
        return _request(model, url, headers, payload)
    except BadRequestError:
        # Degrade gracefully: thinking-ON costs tokens but still answers.
        result = _request(model, url, headers, base_payload)
        _disable_extra_params = True
        print("[degrade] endpoint rejected extra_params (HTTP 400); "
              "dropping them for the rest of the run", file=sys.stderr, flush=True)
        return result


def _request(model, url, headers, payload):
    """POST with transient-error retries. Raises typed errors on 400/404."""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        if attempt:
            time.sleep(BACKOFF_BASE ** (attempt - 1))
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last_err = f"connection error: {e}"
            continue
        if resp.status_code in _RETRYABLE_STATUS:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            continue
        if resp.status_code == 404:
            raise ModelUnavailableError(f"HTTP 404: {resp.text[:300]}")
        if resp.status_code == 400:
            raise BadRequestError(f"HTTP 400: {resp.text[:300]}")
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        usage = dict(data.get("usage", {}))
        _log_usage(model, usage)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"malformed response: {e}: {str(data)[:300]}")
        usage["finish_reason"] = choice.get("finish_reason")
        return (content or "").strip(), usage

    raise RuntimeError(f"exhausted retries: {last_err}")
