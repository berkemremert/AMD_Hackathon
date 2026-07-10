from __future__ import annotations

import os
import threading
import time
from typing import Any

from frugalrouter.prompting import CATEGORY_INSTRUCTIONS, clean_answer
from frugalrouter.types import Answer, Task


LOCAL_MODEL_SYSTEM = (
    "You are a precise assistant. Answer directly and minimally, with no preamble, "
    "no explanation unless asked, and no restating of the question."
)


class LocalModelProvider:
    """A small GGUF model run on-CPU via llama-cpp-python.

    Answers count as ZERO Fireworks tokens (guide: local inference is free),
    so routing easy tasks here is pure token-ranking win. The provider is
    designed to degrade gracefully: if the weights are missing or llama_cpp
    is not installed, ``available_for`` returns False and the router falls
    through to Fireworks exactly as the baseline does. This is why bundling
    the model is optional and the baseline image is unaffected.
    """

    name = "local_model"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("local_model", {})
        # Env overrides config so the same image behaves as baseline when
        # LOCAL_MODEL_PATH is unset.
        self.model_path = os.getenv("LOCAL_MODEL_PATH", cfg.get("model_path", "")).strip()
        self.model_label = cfg.get("model_label", "local-gguf")
        self.categories = set(cfg.get("categories", []))
        self.max_tokens = int(cfg.get("max_tokens", 256))
        self.n_ctx = int(cfg.get("n_ctx", 2048))
        self.n_threads = int(cfg.get("n_threads", os.cpu_count() or 2))
        self._llm: Any = None
        self._load_failed = False
        # llama.cpp is not thread-safe; the CLI runs tasks in parallel workers,
        # so all load + inference on the shared model must be serialized or it
        # segfaults. Fireworks calls don't take this lock and stay parallel.
        self._lock = threading.Lock()

    def available_for(self, category: str) -> bool:
        return category in self.categories and self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._llm is not None:
                return True
            if self._load_failed or not self.model_path or not os.path.exists(self.model_path):
                self._load_failed = True
                return False
            try:
                from llama_cpp import Llama

                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
                return True
            except Exception:  # noqa: BLE001 - any load failure => fall back to remote
                self._load_failed = True
                return False

    def answer(self, task: Task, category: str = "general") -> Answer:
        started = time.perf_counter()
        instruction = CATEGORY_INSTRUCTIONS.get(category, CATEGORY_INSTRUCTIONS["general"])
        with self._lock:
            completion = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                    {"role": "user", "content": f"{instruction}\n{task.input}"},
                ],
                temperature=0.0,
                max_tokens=self.max_tokens,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = completion["choices"][0]
        text = clean_answer((choice.get("message") or {}).get("content") or "", category)
        # prompt/completion tokens stay 0: local inference is free for scoring.
        return Answer(
            text=text,
            provider=self.name,
            model=self.model_label,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )
