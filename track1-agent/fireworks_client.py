"""Thin wrapper around the Fireworks chat completions API.

Reads FIREWORKS_API_KEY / FIREWORKS_BASE_URL from the environment. All model
calls in this project go through here so token usage is tracked in one place,
matching how the hackathon's judging proxy records tokens centrally.
"""
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")


def chat(model: str, prompt: str, max_tokens: int = 800, temperature: float = 0.0, retries: int = 3, api_key: str = None, response_format: dict = None, extra_params: dict = None, system_prompt: str = None) -> dict:
    """Sends one prompt to `model`. Returns {"text": str, "prompt_tokens": int,
    "completion_tokens": int, "total_tokens": int, "finish_reason": str}."""
    url = f"{BASE_URL}/chat/completions"
    key_to_use = api_key or API_KEY
    headers = {"Authorization": f"Bearer {key_to_use}", "Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    if extra_params:
        payload.update(extra_params)
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            choice_obj = data["choices"][0]
            choice = choice_obj["message"]
            text = choice.get("content") or ""
            # Strip <think>...</think> tags entirely to keep response clean and short
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
            
            finish_reason = choice_obj.get("finish_reason")
            usage = data.get("usage", {})
            return {
                "text": text.strip(),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "finish_reason": finish_reason,
            }
        except requests.exceptions.HTTPError as e:
            # If the model doesn't support reasoning params (e.g. Minimax), it returns 400 Bad Request
            if e.response.status_code == 400:
                stripped = False
                for param in ("reasoning_effort", "reasoning_history"):
                    if param in payload:
                        del payload[param]
                        stripped = True
                if stripped:
                    continue  # Retry instantly without the unsupported parameters
            
            last_err = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Fireworks call to {model} failed after {retries} attempts: {last_err}")
