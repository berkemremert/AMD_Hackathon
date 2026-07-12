"""Lazy, thread-safe GLiNER model loader."""
from __future__ import annotations

import sys
import threading
from typing import Any


MODEL_ID = "urchade/gliner_small-v2.1"
_model: Any | None = None
_lock = threading.Lock()


def load_model() -> Any | None:
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model
        try:
            from gliner import GLiNER

            print(f"[GLiNER] Loading {MODEL_ID}...", file=sys.stderr)
            _model = GLiNER.from_pretrained(MODEL_ID)
        except Exception as exc:
            print(f"[GLiNER] Failed to load model: {exc}", file=sys.stderr)
            return None
    return _model
