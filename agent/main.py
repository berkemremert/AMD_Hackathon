"""Entry point for the Category 6 code-debugging agent.

Loads tasks from a JSON file, runs each through the Category6Debugger
pipeline (prompt → local LLM → clean → validate → retry), and writes
results in the competition output format.

Supports two environments:
  - Competition container: reads /input/tasks.json, writes /output/results.json
  - Local development:     reads data/cat6_tasks.json, writes outputs/cat6_local_results.json
"""

import json
from pathlib import Path

from local_llm import LocalLLM
from cat6_debugger import Category6Debugger


def load_tasks() -> list[dict]:
    """Load task list from the competition path or local fallback.

    Returns:
        List of dicts, each containing at least ``task_id`` and ``prompt``.

    Raises:
        FileNotFoundError: If neither the competition nor local task file exists.
    """
    official_path = Path("/input/tasks.json")
    local_path = Path("data/cat6_tasks.json")

    if official_path.exists():
        return json.loads(official_path.read_text())

    if local_path.exists():
        return json.loads(local_path.read_text())

    raise FileNotFoundError("No tasks file found.")


def write_results(results: list[dict]) -> None:
    """Persist results to the competition path or local fallback.

    Args:
        results: List of ``{"task_id": ..., "answer": ...}`` dicts.
    """
    official_output_dir = Path("/output")
    local_output_dir = Path("outputs")

    if official_output_dir.exists():
        output_path = official_output_dir / "results.json"
    else:
        local_output_dir.mkdir(exist_ok=True)
        output_path = local_output_dir / "cat6_local_results.json"

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    """Run the Category 6 debugging pipeline over all loaded tasks."""
    tasks = load_tasks()

    llm = LocalLLM(model="qwen2.5-coder:3b")
    debugger = Category6Debugger(llm=llm, mode="code_only")

    results = []

    for index, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        prompt = task["prompt"]

        print(f"[{index}/{len(tasks)}] Solving {task_id}")

        answer = debugger.solve(prompt)

        results.append({
            "task_id": task_id,
            "answer": answer,
        })

    write_results(results)


if __name__ == "__main__":
    main()
