from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from frugalrouter.types import Task


def read_tasks(path: str | Path) -> list[Task]:
    task_path = Path(path)
    if task_path.suffix.lower() == ".jsonl":
        return read_tasks_jsonl(task_path)
    return read_tasks_json(task_path)


def read_tasks_json(path: str | Path) -> list[Task]:
    task_path = Path(path)
    # utf-8-sig tolerates a UTF-8 BOM, which Windows tools often emit.
    with task_path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Track 1 input must be a JSON array of task objects.")

    tasks: list[Task] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Task at index {index} is not an object.")
        task_id = row.get("task_id") or row.get("id") or f"task-{index}"
        prompt = row.get("prompt") or row.get("input")
        if not prompt:
            raise ValueError(f"Task {task_id} is missing 'prompt'.")
        tasks.append(
            Task(
                id=str(task_id),
                input=str(prompt),
                expected_format=row.get("expected_format"),
                metadata={k: v for k, v in row.items() if k not in {"task_id", "id", "prompt", "input", "expected_format"}},
            )
        )
    return tasks


def read_tasks_jsonl(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            tasks.append(
                Task(
                    id=str(payload.get("id", f"task-{line_number}")),
                    input=str(payload["input"]),
                    expected_format=payload.get("expected_format"),
                    metadata={k: v for k, v in payload.items() if k not in {"id", "input", "expected_format"}},
                )
            )
    return tasks


def write_results(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    if output_path.suffix.lower() == ".jsonl":
        write_results_jsonl(output_path, rows)
    else:
        write_results_json(output_path, rows)


def write_results_json(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    official_rows = []
    for row in rows:
        official_rows.append({"task_id": row["task_id"], "answer": row["answer"]})
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(official_rows, file, ensure_ascii=True, separators=(",", ":"))


def write_results_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")
