from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from frugalrouter.prompting import clean_answer, user_prompt
from frugalrouter.types import Answer, Task


class FireworksError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FireworksProvider:
    name = "fireworks"

    def __init__(self, config: dict[str, Any]) -> None:
        fireworks_config = config.get("fireworks", {})
        self.base_url = fireworks_config.get("base_url", "https://api.fireworks.ai/inference/v1")
        self.default_model = fireworks_config.get("default_model", "gemma-4-31b-it-nvfp4")
        self.temperature = float(fireworks_config.get("temperature", 0.0))
        self.max_tokens = fireworks_config.get("max_tokens", 256)
        # Extra request fields keyed by model-name substring, e.g. turning
        # off Gemma's thinking mode with {"reasoning_effort": "none"}.
        self.model_overrides = config.get("model_overrides", {})
        self.api_key = os.getenv("FIREWORKS_API_KEY")

    def answer(
        self,
        task: Task,
        model: str | None = None,
        category: str = "general",
        max_tokens_override: int | None = None,
        no_reasoning_directive: bool = False,
        skip_model_overrides: bool = False,
    ) -> Answer:
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY is not set.")

        selected_model = model or self.default_model
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "user", "content": user_prompt(task, category, no_reasoning=no_reasoning_directive)},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens_override or self._max_tokens(category),
        }
        # V12 applies compact-reasoning controls to every category. Math and
        # logic still receive task-specific instructions, but their hidden
        # reasoning cannot consume the leaderboard's full token budget.
        if not skip_model_overrides:
            payload.update(self._request_overrides(selected_model, category))

        request = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=28) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise FireworksError(f"Fireworks HTTP {error.code}: {detail}", status_code=error.code) from error

        latency_ms = int((time.perf_counter() - started) * 1000)
        first_choice = body["choices"][0]
        # Reasoning models can hit the token cap before emitting any final
        # content, leaving message without a "content" key.
        content = (first_choice.get("message") or {}).get("content") or ""
        choice = clean_answer(content, category, prompt=task.input)
        usage = body.get("usage", {})
        return Answer(
            text=choice,
            provider=self.name,
            model=selected_model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            finish_reason=first_choice.get("finish_reason"),
        )

    def _max_tokens(self, category: str) -> int:
        if isinstance(self.max_tokens, dict):
            return int(self.max_tokens.get(category, self.max_tokens.get("general", 220)))
        return int(self.max_tokens)

    def _request_overrides(self, selected_model: str, category: str) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for pattern, extra in self.model_overrides.items():
            if pattern.lower() not in selected_model.lower():
                continue
            for field, value in extra.items():
                if isinstance(value, dict):
                    category_value = value.get(category, value.get("default"))
                    if category_value is not None:
                        resolved[field] = category_value
                else:
                    resolved[field] = value
        return resolved

    def _chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"
