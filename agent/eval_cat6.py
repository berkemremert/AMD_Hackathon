"""Local evaluation harness for the Category 6 debugging pipeline.

Runs both ``code_only`` and ``bug_plus_code`` modes over the synthetic
Category 6 dataset, saves per-task results, and prints a summary of
key metrics (syntax validity, empty answers, stray markdown fences).
"""

import json
from pathlib import Path

from local_llm import LocalLLM
from cat6_debugger import Category6Debugger
from validators import detect_language, validate_code

EVAL_DATA_PATH = Path("data/category6_synthetic_v1.json")


def evaluate_mode(tasks: list[dict], mode: str) -> list[dict]:
    """Run all *tasks* through the Category6Debugger in the given *mode*.

    Args:
        tasks: List of task dicts (``task_id``, ``prompt``).
        mode: ``"code_only"`` or ``"bug_plus_code"``.

    Returns:
        List of result dicts including answer, language, and validity info.
    """
    llm = LocalLLM(model="qwen2.5-coder:3b")
    debugger = Category6Debugger(llm=llm, mode=mode)

    results = []

    for i, task in enumerate(tasks, start=1):
        print(f"[{mode}] [{i}/{len(tasks)}] {task['task_id']}")
        answer = debugger.solve(task["prompt"])
        language = detect_language(task["prompt"])
        is_valid, error = validate_code(answer, language)

        results.append({
            "task_id": task["task_id"],
            "answer": answer,
            "language": language,
            "syntax_valid": is_valid,
            "syntax_error": error,
            "answer_length": len(answer),
        })

    return results


def summarize(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation *results*.

    Returns:
        Dict with counts for total tasks, valid syntax, empty answers,
        stray markdown fences, and average answer length.
    """
    total = len(results)
    return {
        "total_tasks": total,
        "syntax_valid_count": sum(1 for r in results if r["syntax_valid"]),
        "empty_answer_count": sum(1 for r in results if not r["answer"].strip()),
        "markdown_fence_count": sum(1 for r in results if "```" in r["answer"]),
        "avg_answer_length": sum(r["answer_length"] for r in results) / max(total, 1),
    }


def main():
    """Run both evaluation modes and save results + summaries."""
    tasks = json.loads(EVAL_DATA_PATH.read_text())

    Path("outputs").mkdir(exist_ok=True)

    for mode in ["code_only", "bug_plus_code"]:
        results = evaluate_mode(tasks, mode)
        summary = summarize(results)

        Path(f"outputs/cat6_{mode}_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )

        Path(f"outputs/cat6_{mode}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )

        print(mode, summary)


if __name__ == "__main__":
    main()
