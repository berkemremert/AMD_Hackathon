"""Evaluate the production task processor on the public 80-question set."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fireworks_client import chat
from src.track1_agent.config import load_settings
from src.track1_agent.pipeline import ProcessResult, TaskProcessor


DATA_PATH = Path(__file__).parent / "data" / "input" / "public_style_80_questions.json"
JUDGE_MODEL = os.environ.get("MODEL_JUDGE", "accounts/fireworks/models/glm-5p2")


def judge(prompt: str, answer: str) -> dict:
    judge_prompt = (
        "Decide whether the candidate answer is correct and complete.\n\n"
        f"Task:\n{prompt}\n\nCandidate answer:\n{answer}\n\n"
        "Reply with one line: CORRECT or INCORRECT, followed by a short reason."
    )
    try:
        response = chat(
            model=JUDGE_MODEL,
            prompt=judge_prompt,
            max_tokens=60,
            system_prompt="Be a strict answer verifier. Return one concise verdict line.",
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"},
        )
    except Exception as exc:
        return {"verdict": "error", "reason": str(exc), "tokens": 0}

    text = response["text"].strip()
    return {
        "verdict": "correct" if text.upper().startswith("CORRECT") else "incorrect",
        "reason": text,
        "tokens": response["total_tokens"],
    }


def _empty_category_stats() -> dict:
    return {
        "total_questions": 0,
        "local_count": 0,
        "api_count": 0,
        "tokens_used": 0,
        "retries": 0,
        "judge_correct": 0,
        "judge_total": 0,
    }


def _entry(task: dict, result: ProcessResult, verdict: dict) -> dict:
    return {
        "task_id": task["id"],
        "category_dataset": task["category"],
        "category_detected": result.task_type,
        "prompt": task["prompt"],
        "solver_type": "local" if result.source.startswith("local:") else "api",
        "model_or_solver": result.source if result.model is None else result.model,
        "tokens_used": result.tokens,
        "output": result.answer,
        "finish_reason": result.finish_reason,
        "judge_verdict": verdict["verdict"],
        "judge_reason": verdict["reason"],
        "judge_tokens": verdict["tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the production task processor.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATA_PATH),
        help="Path to the dataset JSON file to use."
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        help="List of categories to test. If not provided, all categories are tested."
    )
    parser.add_argument(
        "--questions-per-category",
        type=int,
        default=None,
        help="How many questions per category should be selected randomly."
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    all_tasks = json.loads(dataset_path.read_text())

    output_dir = Path(__file__).parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"eval_results_{dataset_path.stem}_{timestamp}.json"

    # Normalize tasks to always have 'id' and 'category'
    for t in all_tasks:
        if "id" not in t and "task_id" in t:
            t["id"] = t["task_id"]
        if "category" not in t and "id" in t:
            # Infer category from id prefix (e.g., 'factual_01' -> 'factual')
            t["category"] = t["id"].split("_")[0] if "_" in t["id"] else "unknown"

    if args.categories:
        allowed_categories = set(args.categories)
        all_tasks = [t for t in all_tasks if t.get("category") in allowed_categories]

    if args.questions_per_category is not None:
        tasks_by_category = defaultdict(list)
        for t in all_tasks:
            tasks_by_category[t.get("category")].append(t)
        
        tasks = []
        for cat, cat_tasks in tasks_by_category.items():
            if len(cat_tasks) > args.questions_per_category:
                tasks.extend(random.sample(cat_tasks, args.questions_per_category))
            else:
                tasks.extend(cat_tasks)
    else:
        tasks = all_tasks

    processor = TaskProcessor(load_settings().api_model)
    entries: list[dict] = []
    judge_counts = defaultdict(int)
    total_api_tokens = 0
    total_judge_tokens = 0

    for index, task in enumerate(tasks, 1):
        print(f"[{index:02}/{len(tasks)}] {task['id']} — {task['category']}")
        try:
            result = processor.process(task["prompt"])
        except Exception as exc:
            result = ProcessResult(
                answer=f"Unable to process task: {exc}",
                task_type="error",
                source="error",
            )

        verdict = judge(task["prompt"], result.answer)
        entry = _entry(task, result, verdict)
        entries.append(entry)
        total_api_tokens += result.tokens
        total_judge_tokens += verdict["tokens"]
        judge_counts[verdict["verdict"]] += 1
        print(
            f"  {result.source} | tokens={result.tokens} | "
            f"judge={verdict['verdict']} | {result.answer[:120]!r}"
        )

    category_breakdown: dict[str, dict] = {}
    for entry in entries:
        stats = category_breakdown.setdefault(entry["category_dataset"], _empty_category_stats())
        stats["total_questions"] += 1
        stats["tokens_used"] += entry["tokens_used"]
        if entry["solver_type"] == "local":
            stats["local_count"] += 1
        else:
            stats["api_count"] += 1
        if entry["judge_verdict"] in {"correct", "incorrect"}:
            stats["judge_total"] += 1
            stats["judge_correct"] += int(entry["judge_verdict"] == "correct")

    judged = judge_counts["correct"] + judge_counts["incorrect"]
    accuracy = 100 * judge_counts["correct"] / judged if judged else 0.0
    judge_results = {
        "correct": judge_counts["correct"],
        "incorrect": judge_counts["incorrect"],
        "error": judge_counts["error"],
    }
    total_local = sum(stats["local_count"] for stats in category_breakdown.values())
    output = {
        "summary": {
            "total_tasks": len(tasks),
            "success_count": sum(entry["category_detected"] != "error" for entry in entries),
            "total_local_tasks": total_local,
            "total_api_tasks": len(tasks) - total_local,
            "total_retries": 0,
            "total_api_tokens": total_api_tokens,
            "approximate_score": total_api_tokens / len(tasks) * 19,
            "judge_accuracy_pct": round(accuracy, 1),
            "judge_results": judge_results,
            "judge_tokens": total_judge_tokens,
            "category_breakdown": category_breakdown,
        },
        "results": entries,
    }
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print("\n" + "-" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Total API Tokens Used: {total_api_tokens}")
    print(f"Tasks Successfully Processed: {output['summary']['success_count']}/{len(tasks)}")
    print(f"Total Tasks Handled Locally (0 tokens): {total_local}/{len(tasks)}")
    print("Total Retries: 0")
    print(f"Approximate score: {output['summary']['approximate_score']:.2f}")
    print("Category Breakdown (Questions / Local / API / Tokens / Retries / Accuracy):")
    for category, stats in sorted(category_breakdown.items()):
        correct = stats["judge_correct"]
        total = stats["judge_total"]
        pct = 100 * correct / total if total else 0
        print(
            f"  • {category:<26} | Total: {stats['total_questions']:<2} "
            f"| Local: {stats['local_count']:<2} | API: {stats['api_count']:<2} "
            f"| Tokens: {stats['tokens_used']:<5} | Retries: 0 "
            f"| Acc: {pct:.0f}% ({correct}/{total})"
        )
    print("-" * 80)
    print(
        f"Judge: ✓ {judge_counts['correct']} correct | "
        f"⚠ {judge_counts['incorrect']} incorrect | ✗ {judge_counts['error']} errors"
    )


if __name__ == "__main__":
    main()
