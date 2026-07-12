"""Runtime configuration loaded from the judging environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    input_path: Path
    output_path: Path
    api_model: str


def select_kimi_model() -> str:
    """Select a Kimi model without ever inventing an allowlisted ID."""
    raw_allowed = os.environ.get("ALLOWED_MODELS", "")
    allowed = tuple(item.strip() for item in raw_allowed.split(",") if item.strip())

    if not allowed:
        return os.environ.get("MODEL_API", "accounts/fireworks/models/kimi-k2p6")

    kimi_models = tuple(model for model in allowed if "kimi" in model.lower())
    if not kimi_models:
        raise RuntimeError("ALLOWED_MODELS does not contain a Kimi model")

    override = os.environ.get("MODEL_API")
    return override if override in kimi_models else kimi_models[0]


def load_settings() -> Settings:
    return Settings(
        input_path=Path(os.environ.get("TASK_INPUT_PATH", "/input/tasks.json")),
        output_path=Path(os.environ.get("TASK_OUTPUT_PATH", "/output/results.json")),
        api_model=select_kimi_model(),
    )
