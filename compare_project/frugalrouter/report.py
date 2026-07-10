from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FrugalRouter decision logs.")
    parser.add_argument("--log", default="logs/decisions.jsonl", help="Decision log JSONL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"No decision log found at {path}")

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))

    if not rows:
        raise SystemExit("Decision log is empty.")

    remote_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    confidences = []
    routes: dict[str, int] = {}

    for row in rows:
        result = row["result"]
        answer = result["answer"]
        route = result["route"]
        routes[route] = routes.get(route, 0) + 1
        remote_calls += 1 if result["used_remote"] else 0
        prompt_tokens += int(answer.get("prompt_tokens", 0))
        completion_tokens += int(answer.get("completion_tokens", 0))
        confidences.append(float(result["verification"]["confidence"]))

    total = len(rows)
    avg_confidence = sum(confidences) / len(confidences)

    print(f"Tasks: {total}")
    print(f"Remote calls: {remote_calls} ({remote_calls / total:.1%})")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Completion tokens: {completion_tokens}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print("Routes:")
    for route, count in sorted(routes.items()):
        print(f"  {route}: {count}")


if __name__ == "__main__":
    main()

