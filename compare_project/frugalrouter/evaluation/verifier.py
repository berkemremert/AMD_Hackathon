from __future__ import annotations

import re

from frugalrouter.types import Answer, Task, Verification


class LocalVerifier:
    def score(self, task: Task, answer: Answer) -> Verification:
        confidence = 0.8 if answer.provider == "fireworks" else 0.0
        reasons: list[str] = []
        text = answer.text.strip()

        if not text:
            return Verification(confidence=0.0, reasons=["empty_answer"])

        if task.expected_format == "label":
            allowed = {"positive", "neutral", "negative"}
            if text.lower().split(":", 1)[0].strip() in allowed:
                confidence += 0.1
                reasons.append("valid_label")
            else:
                confidence -= 0.25
                reasons.append("invalid_label")

        if task.expected_format == "short_text":
            word_count = len(re.findall(r"\S+", text))
            if word_count <= 40:
                confidence += 0.05
                reasons.append("concise_answer")
            else:
                confidence -= 0.1
                reasons.append("long_answer")

        if self._looks_uncertain(text):
            confidence -= 0.25
            reasons.append("uncertain_language")

        return Verification(confidence=max(0.0, min(1.0, confidence)), reasons=reasons)

    def _looks_uncertain(self, text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in ["i don't know", "maybe", "not sure", "cannot determine"])
