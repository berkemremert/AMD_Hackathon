from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from frugalrouter.config import env_flag, load_config
from frugalrouter.decision_log import DecisionLogger
from frugalrouter.io import read_tasks, write_results
from frugalrouter.router import FrugalRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FrugalRouter on Track 1 tasks.")
    parser.add_argument("--tasks", default="/input/tasks.json", help="Input JSON or JSONL task file.")
    parser.add_argument("--output", default="/output/results.json", help="Output JSON or JSONL result file.")
    parser.add_argument("--config", default="config/models.json", help="Router config JSON.")
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Disable Fireworks calls for local shortcut debugging only.",
    )
    parser.add_argument(
        "--debug-output",
        action="store_true",
        help="Include route/debug fields. Do not use for official submission output.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel task workers.")
    parser.add_argument("--decision-log", default=None, help="Decision log JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    allow_remote = not args.no_remote and env_flag("FRUGAL_ALLOW_REMOTE", default=True)
    router = FrugalRouter(config=config, allow_remote=allow_remote)
    logger = DecisionLogger(args.decision_log)
    tasks = read_tasks(args.tasks)
    workers = args.workers or int(os.getenv("FRUGAL_WORKERS", "4"))

    if workers <= 1 or len(tasks) <= 1:
        results = [router.run(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(router.run, tasks))

    rows = []
    remote_count = sum(1 for result in results if result.used_remote)
    for task, result in zip(tasks, results, strict=True):
        logger.write(task, result)
        row = {
            "task_id": result.task_id,
            "answer": result.answer.text,
        }
        if args.debug_output:
            row.update(
                {
                    "route": result.route,
                    "category": result.category,
                    "confidence": round(result.verification.confidence, 4),
                    "used_remote": result.used_remote,
                    "fallback_reason": result.fallback_reason,
                    "usage": {
                        "prompt_tokens": result.answer.prompt_tokens,
                        "completion_tokens": result.answer.completion_tokens,
                        "latency_ms": result.answer.latency_ms,
                        "model": result.answer.model,
                        "provider": result.answer.provider,
                    },
                    "debug": asdict(result.verification),
                }
            )
        rows.append(row)

    write_results(args.output, rows)
    print(f"Wrote {len(rows)} results to {args.output}. Remote calls: {remote_count}.")


if __name__ == "__main__":
    main()
