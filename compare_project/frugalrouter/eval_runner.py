from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from frugalrouter.config import env_flag, load_config
from frugalrouter.decision_log import DecisionLogger
from frugalrouter.io import read_tasks
from frugalrouter.router import FrugalRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local FrugalRouter evaluation set.")
    parser.add_argument("--tasks", default="data/eval_tasks.json", help="Eval task JSON file.")
    parser.add_argument("--config", default="config/models.json", help="Router config JSON.")
    parser.add_argument("--out-dir", default="reports", help="Directory for eval artifacts.")
    parser.add_argument("--no-remote", action="store_true", help="Disable Fireworks calls.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel task workers.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between tasks (sequential runs only); paces personal-key rate limits.",
    )
    parser.add_argument("--decision-log", default=None, help="Decision log JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    task_specs = _load_specs(args.tasks)
    if args.limit is not None:
        task_specs = task_specs[: args.limit]

    config = load_config(args.config)
    allow_remote = not args.no_remote and env_flag("FRUGAL_ALLOW_REMOTE", default=True)
    router = FrugalRouter(config=config, allow_remote=allow_remote)
    logger = DecisionLogger(args.decision_log)
    tasks = read_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    workers = args.workers or int(os.getenv("FRUGAL_WORKERS", "4"))
    if workers <= 1 or len(tasks) <= 1:
        results = []
        for index, task in enumerate(tasks):
            if index and args.delay > 0:
                time.sleep(args.delay)
            results.append(router.run(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(router.run, tasks))

    rows = []
    for spec, task, result in zip(task_specs, tasks, results, strict=True):
        logger.write(task, result)
        grade = grade_answer(result.answer.text, spec)
        rows.append(
            {
                "task_id": result.task_id,
                "expected_category": spec.get("category"),
                "detected_category": result.category,
                "answer": result.answer.text,
                "passed": grade["passed"],
                "grade_reason": grade["reason"],
                "route": result.route,
                "used_remote": result.used_remote,
                "model": result.answer.model,
                "provider": result.answer.provider,
                "prompt_tokens": result.answer.prompt_tokens,
                "completion_tokens": result.answer.completion_tokens,
                "finish_reason": result.answer.finish_reason,
                "latency_ms": result.answer.latency_ms,
                "fallback_reason": result.fallback_reason,
                "verification": asdict(result.verification),
            }
        )

    report = summarize(rows, elapsed_seconds=time.perf_counter() - started)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "eval_results.json", rows)
    _write_json(out_dir / "eval_report.json", report)
    print_report(report)


def grade_answer(answer: str, spec: dict[str, Any]) -> dict[str, Any]:
    normalized_answer = normalize(answer)
    primary = _grade_primary(answer, normalized_answer, spec)

    # expected_max_words is an optional additional constraint that ANDs with
    # the primary grader, so a task can require correct content AND a length
    # limit (tests format obedience, which bears on strict grading + tokens).
    if primary["passed"] and "expected_max_words" in spec:
        word_count = len(re.findall(r"\S+", answer.strip()))
        limit = int(spec["expected_max_words"])
        if word_count > limit:
            return _grade(False, f"{primary['reason']};over_word_limit:{word_count}>{limit}")

    return primary


def _grade_primary(answer: str, normalized_answer: str, spec: dict[str, Any]) -> dict[str, Any]:
    if "expected_exact" in spec:
        expected = normalize(str(spec["expected_exact"]))
        return _grade(normalized_answer == expected, f"exact:{spec['expected_exact']}")

    if "expected_label" in spec:
        expected = normalize(str(spec["expected_label"]))
        first_token = normalized_answer.split(":", 1)[0].split(".", 1)[0].strip()
        return _grade(first_token == expected, f"label:{spec['expected_label']}")

    if "expected_regex" in spec:
        pattern = str(spec["expected_regex"])
        return _grade(re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE) is not None, f"regex:{pattern}")

    if "expected_contains_all" in spec:
        missing = [item for item in spec["expected_contains_all"] if normalize(str(item)) not in normalized_answer]
        return _grade(not missing, f"contains_all_missing:{missing}")

    if "expected_contains_any" in spec:
        matched = [item for item in spec["expected_contains_any"] if normalize(str(item)) in normalized_answer]
        return _grade(bool(matched), f"contains_any_matched:{matched}")

    if "expected_max_words" in spec:
        return _grade(True, "word_limit_only")

    return {"passed": None, "reason": "ungraded"}


def summarize(rows: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    graded = [row for row in rows if row["passed"] is not None]
    passed = [row for row in graded if row["passed"]]
    remote = [row for row in rows if row["used_remote"]]
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
    completion_tokens = sum(int(row["completion_tokens"]) for row in rows)

    routes: dict[str, int] = {}
    categories: dict[str, int] = {}
    for row in rows:
        routes[row["route"]] = routes.get(row["route"], 0) + 1
        categories[row["detected_category"]] = categories.get(row["detected_category"], 0) + 1

    with_expected = [row for row in rows if row.get("expected_category")]
    matched = [row for row in with_expected if row["expected_category"] == row["detected_category"]]

    local_graded = [row for row in graded if row["route"] == "local"]
    local_wrong = [row for row in local_graded if not row["passed"]]

    truncated = [row for row in rows if row.get("finish_reason") == "length"]

    per_category: dict[str, dict[str, Any]] = {}
    for row in graded:
        bucket = per_category.setdefault(row.get("expected_category") or "unknown", {"graded": 0, "passed": 0})
        bucket["graded"] += 1
        bucket["passed"] += 1 if row["passed"] else 0
    for bucket in per_category.values():
        bucket["accuracy"] = round(bucket["passed"] / bucket["graded"], 3) if bucket["graded"] else None

    return {
        "tasks": len(rows),
        "graded_tasks": len(graded),
        "passed": len(passed),
        "accuracy": (len(passed) / len(graded)) if graded else None,
        "remote_calls": len(remote),
        "remote_rate": (len(remote) / len(rows)) if rows else 0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "routes": routes,
        "categories": categories,
        "classification_accuracy": (len(matched) / len(with_expected)) if with_expected else None,
        "misclassified_task_ids": [row["task_id"] for row in with_expected if row not in matched],
        "local_answered": len(local_graded),
        "local_passed": sum(1 for row in local_graded if row["passed"]),
        "local_wrong_task_ids": [row["task_id"] for row in local_wrong],
        "truncated_count": len(truncated),
        "truncated_task_ids": [row["task_id"] for row in truncated],
        "per_category": per_category,
        "failed_task_ids": [row["task_id"] for row in graded if not row["passed"]],
    }


def print_report(report: dict[str, Any]) -> None:
    accuracy = report["accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.1%}"
    print(f"Tasks: {report['tasks']}")
    print(f"Graded accuracy: {accuracy_text} ({report['passed']}/{report['graded_tasks']})")
    print(f"Remote calls: {report['remote_calls']} ({report['remote_rate']:.1%})")
    print(f"Tokens: {report['total_tokens']} total ({report['prompt_tokens']} prompt, {report['completion_tokens']} completion)")
    print(f"Elapsed: {report['elapsed_seconds']}s")
    print(f"Routes: {report['routes']}")
    print(f"Categories: {report['categories']}")
    classification = report["classification_accuracy"]
    if classification is not None:
        print(f"Classification accuracy: {classification:.1%}")
    if report["misclassified_task_ids"]:
        print(f"Misclassified: {', '.join(report['misclassified_task_ids'])}")
    print(f"Local answered: {report['local_answered']} (wrong: {len(report['local_wrong_task_ids'])})")
    if report["local_wrong_task_ids"]:
        print(f"!! LOCAL WRONG (accuracy-gate risk): {', '.join(report['local_wrong_task_ids'])}")
    if report["truncated_count"]:
        print(f"!! Truncated (finish_reason=length): {', '.join(report['truncated_task_ids'])}")
    if report["failed_task_ids"]:
        print(f"Failed: {', '.join(report['failed_task_ids'])}")


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _grade(passed: bool, reason: str) -> dict[str, Any]:
    return {"passed": passed, "reason": reason}


def _load_specs(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Eval tasks must be a JSON array.")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()
