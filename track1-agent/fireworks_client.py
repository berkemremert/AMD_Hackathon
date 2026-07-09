"""Thin wrapper around the Fireworks chat completions API.

Reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL from the environment. All model
calls in this project go through here so token usage is tracked in one place,
matching how the hackathon's judging proxy records tokens centrally.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")


def chat(model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.0, retries: int = 3) -> dict:
    """Sends one prompt to `model`. Returns {"text": str, "prompt_tokens": int,
    "completion_tokens": int, "total_tokens": int}."""
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            text = choice.get("content") or ""
            usage = data.get("usage", {})
            return {
                "text": text.strip(),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Fireworks call to {model} failed after {retries} attempts: {last_err}")
