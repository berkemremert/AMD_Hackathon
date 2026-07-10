from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    input: str
    expected_format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    finish_reason: str | None = None


@dataclass(frozen=True)
class Verification:
    confidence: float
    reasons: list[str]


@dataclass(frozen=True)
class RouteResult:
    task_id: str
    answer: Answer
    verification: Verification
    route: str
    used_remote: bool
    fallback_reason: str | None = None
    category: str = "general"
