"""Track 1 entrypoint: read /input/tasks.json, answer each task, write
/output/results.json, exit 0.

The judging harness mounts /input and /output at the filesystem root and
injects FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS. Paths are
overridable via INPUT_PATH / OUTPUT_PATH for local development.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from agent import solve
from llm import describe_tiers, usage

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
# Stop with headroom before the harness's 10-minute kill so results.json is
# always written, even if a few tasks never come back.
DEADLINE_S = float(os.environ.get("DEADLINE_S", "480"))


def load_tasks(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        tasks = json.load(fh)
    if not isinstance(tasks, list):
        raise ValueError(f"expected a JSON list, got {type(tasks).__name__}")
    return tasks


def write_results(path: str, results: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)


def _answer_one(task: dict, index: int) -> dict:
    # Echo task_id exactly as given (numbers stay numbers); fabricate a
    # stable one only when the input omits it.
    task_id = task.get("task_id", f"idx_{index}")
    try:
        answer = solve(task.get("prompt", ""))
    except Exception:
        traceback.print_exc()
        answer = ""
    return {"task_id": task_id, "answer": answer}


def run(tasks: list[dict]) -> list[dict]:
    if len(tasks) <= 1:
        return [_answer_one(t, i) for i, t in enumerate(tasks)]

    deadline = time.monotonic() + DEADLINE_S
    pool = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks)))
    futures = [pool.submit(_answer_one, t, i) for i, t in enumerate(tasks)]

    results: list[dict] = []
    for i, fut in enumerate(futures):
        try:
            results.append(fut.result(timeout=max(1.0, deadline - time.monotonic())))
        except Exception:  # deadline hit: blank answer, keep the id present
            results.append({"task_id": tasks[i].get("task_id", f"idx_{i}"), "answer": ""})
    pool.shutdown(wait=False, cancel_futures=True)
    return results


def main() -> int:
    missing = [k for k in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS")
               if not os.environ.get(k)]
    if missing:
        print(f"FATAL: missing environment variables: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as exc:
        print(f"FATAL: cannot read tasks from {INPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(tasks)} task(s) from {INPUT_PATH}", file=sys.stderr)
    try:
        print(f"Model tiers: {describe_tiers()}", file=sys.stderr)
    except Exception as exc:
        print(f"WARN: could not resolve model tiers: {exc}", file=sys.stderr)

    results = run(tasks)

    try:
        write_results(OUTPUT_PATH, results)
    except Exception as exc:
        print(f"FATAL: cannot write results to {OUTPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    u = usage()
    print(f"Wrote {len(results)} result(s) to {OUTPUT_PATH} | tokens: total={u['total']} "
          f"(prompt={u['prompt']} completion={u['completion']}) over {u['calls']} call(s)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
