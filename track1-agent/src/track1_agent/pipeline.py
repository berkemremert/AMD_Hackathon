"""Single production pipeline shared by the container and evaluator."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable

from fireworks_client import chat
from local_compressor import optimize_prompt_for_api
from output_optimizer import detect_task_type, get_dynamic_limits

from .config import Settings, load_settings
from .solvers import solve_logic_puzzle, solve_math_exact, solve_ner, solve_sentiment


@dataclass(frozen=True)
class ProcessResult:
    answer: str
    task_type: str
    source: str
    model: str | None = None
    tokens: int = 0
    finish_reason: str | None = None


LocalSolver = Callable[[str], str | None]
ApiChat = Callable[..., dict]

LOCAL_SOLVERS: dict[str, tuple[str, LocalSolver]] = {
    "math_solving": ("math", solve_math_exact),
    "logical_puzzles": ("logic", solve_logic_puzzle),
    "entity_extraction": ("ner", solve_ner),
    "sentiment_analysis": ("sentiment", solve_sentiment),
}


class TaskProcessor:
    """Route one prompt to a local solver or one Kimi API call."""

    def __init__(self, model: str, api_chat: ApiChat = chat):
        self.model = model
        self.api_chat = api_chat

    def process(self, prompt: str) -> ProcessResult:
        task_type = detect_task_type(prompt)
        local_result = self._try_local(task_type, prompt)
        if local_result is not None:
            return local_result

        limits = get_dynamic_limits(task_type, prompt)
        optimized_prompt = optimize_prompt_for_api(prompt, task_type)
        response = self.api_chat(
            model=self.model,
            prompt=optimized_prompt,
            max_tokens=limits["cap"],
            system_prompt=limits["system"],
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
        return ProcessResult(
            answer=str(response.get("text", "")),
            task_type=task_type,
            source="api",
            model=self.model,
            tokens=int(response.get("total_tokens", 0)),
            finish_reason=response.get("finish_reason"),
        )

    @staticmethod
    def _try_local(task_type: str, prompt: str) -> ProcessResult | None:
        configured = LOCAL_SOLVERS.get(task_type)
        if configured is None:
            return None

        name, solver = configured
        try:
            answer = solver(prompt)
        except Exception as exc:
            print(f"[WARN] Local {name} solver failed: {exc}. Using Kimi.", file=sys.stderr)
            return None

        if answer is None:
            return None
        return ProcessResult(
            answer=str(answer),
            task_type=task_type,
            source=f"local:{name}",
        )


def _validate_tasks(payload: object) -> list[dict]:
    if not isinstance(payload, list):
        raise ValueError("tasks.json must contain a JSON array")
    for index, task in enumerate(payload):
        if not isinstance(task, dict) or "task_id" not in task or "prompt" not in task:
            raise ValueError(f"task at index {index} must contain task_id and prompt")
        if not isinstance(task["prompt"], str):
            raise ValueError(f"task {task['task_id']} prompt must be a string")
    return payload


def run(
    settings: Settings | None = None,
    processor: TaskProcessor | None = None,
) -> list[dict]:
    settings = settings or load_settings()
    tasks = _validate_tasks(json.loads(settings.input_path.read_text()))
    processor = processor or TaskProcessor(settings.api_model)
    results: list[dict] = []
    total_tokens = 0

    for task in tasks:
        try:
            processed = processor.process(task["prompt"])
            total_tokens += processed.tokens
            answer = processed.answer
        except Exception as exc:
            print(f"[ERROR] Task {task['task_id']} failed: {exc}", file=sys.stderr)
            answer = "Unable to process task."
        results.append({"task_id": task["task_id"], "answer": answer})

    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.output_path.write_text(json.dumps(results, indent=2))
    print(
        f"Wrote {len(results)} results to {settings.output_path}. API tokens: {total_tokens}",
        file=sys.stderr,
    )
    return results
