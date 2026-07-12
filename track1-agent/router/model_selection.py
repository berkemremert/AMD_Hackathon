"""Resolve the single API model strictly from harness-allowed IDs."""
from __future__ import annotations

import os


def resolve_model_roles() -> tuple[str, str]:
    """Return Kimi for both legacy model roles.

    The two-value return shape is retained so existing callers do not need to
    change, but all API paths use the same model. In the judging environment
    the returned ID always comes directly from ``ALLOWED_MODELS``.
    """
    raw_allowed = os.environ.get("ALLOWED_MODELS", "")
    allowed = tuple(model.strip() for model in raw_allowed.split(",") if model.strip())

    if not allowed:
        model = os.environ.get("MODEL_API", "accounts/fireworks/models/kimi-k2p6")
        return model, model

    kimi_models = tuple(model for model in allowed if "kimi" in model.lower())
    if not kimi_models:
        raise RuntimeError("ALLOWED_MODELS does not contain a Kimi model")
    override = os.environ.get("MODEL_API")
    model = override if override in kimi_models else None
    model = model or kimi_models[0]
    return model, model
