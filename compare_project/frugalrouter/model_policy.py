from __future__ import annotations

import re
from typing import Any


DEFAULT_ALLOWED_MODELS = [
    "minimax-m3",
    "kimi-k2p7-code",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it-nvfp4",
]


class ModelPolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.allowed_models = config.get("allowed_models") or DEFAULT_ALLOWED_MODELS
        self.policy = config.get("model_policy", {})
        self.default_model = config.get("fireworks", {}).get("default_model")
        self.include_all_allowed_fallbacks = bool(config.get("include_all_allowed_fallbacks", True))
        self.logic_specialist_patterns = config.get("logic_specialist_patterns", [])

    def choose(self, category: str, prompt: str | None = None) -> str:
        return self.candidates(category, prompt=prompt)[0]

    def candidates(self, category: str, prompt: str | None = None) -> list[str]:
        policy_key = category
        if category == "logic" and prompt and any(
            re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in self.logic_specialist_patterns
        ):
            policy_key = "logic_specialist"
        preferences = self.policy.get(policy_key) or self.policy.get(category) or self.policy.get("general") or []
        if self.default_model:
            preferences = [*preferences, self.default_model]
        if self.include_all_allowed_fallbacks:
            preferences = [*preferences, *DEFAULT_ALLOWED_MODELS]

        candidates: list[str] = []
        for preference in preferences:
            match = self._find_allowed(preference)
            if match and match not in candidates:
                candidates.append(match)

        if not self.allowed_models:
            raise RuntimeError("No allowed Fireworks models are configured.")
        if self.include_all_allowed_fallbacks:
            for model in self.allowed_models:
                if model not in candidates:
                    candidates.append(model)
        if not candidates:
            raise RuntimeError("None of the configured model preferences are allowed by the harness.")
        return candidates

    def _find_allowed(self, preference: str) -> str | None:
        preferred = preference.lower()
        for model in self.allowed_models:
            candidate = model.lower()
            if candidate == preferred or preferred in candidate:
                return model
        return None
