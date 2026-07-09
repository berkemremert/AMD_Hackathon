"""
Runner — runs the agent in different modes and compares them.

Run from the project root:
    python src/runner.py

Part 3: compares three strategies
  - "remote": every task -> paid remote model   (high accuracy, high tokens)
  - "local":  every task -> free local model     (zero tokens, lower accuracy)
  - "router": try local, escalate only if unsure (aim: near-remote accuracy,
              far fewer tokens)

"remote calls" = how many times we hit the PAID model. Fewer is cheaper.
"""

import json
import os

from tokens import TokenLedger
from agent import run_agent
from scoring import is_correct

HERE = os.path.dirname(__file__)
TASKS_PATH = os.path.join(HERE, "..", "tasks", "fake_tasks.json")

THRESHOLD = 0.6   # the router's confidence cutoff (Part 4 will tune this)
SAMPLES = 5       # how many times the router asks the local model


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_mode(tasks, mode):
    """Run every task in one mode. Returns (correct_count, remote_calls, ledger)."""
    ledger = TokenLedger()
    correct = 0
    for task in tasks:
        answer = run_agent(task, ledger, mode=mode, threshold=THRESHOLD, samples=SAMPLES)
        if is_correct(answer, task["answer"]):
            correct += 1
    remote_calls = sum(1 for c in ledger.calls if c["where"] == "remote")
    return correct, remote_calls, ledger


def main():
    tasks = load_tasks(TASKS_PATH)
    total = len(tasks)

    print(f"Comparing strategies on {total} tasks (router threshold = {THRESHOLD})\n")
    header = f"{'mode':<10}{'accuracy':<16}{'remote calls':<16}{'remote tokens (cost)':<22}"
    print(header)
    print("-" * len(header))

    for mode in ("remote", "local", "router"):
        correct, remote_calls, ledger = run_mode(tasks, mode)
        acc = f"{correct}/{total} ({100 * correct // total}%)"
        calls = f"{remote_calls}/{total}"
        print(f"{mode:<10}{acc:<16}{calls:<16}{ledger.remote_total:<22}")

    print("\nThe router aims to match 'remote' accuracy while paying for far fewer tasks.")
    print("Numbers shift each run (mock models are random); the pattern is the point.")


if __name__ == "__main__":
    main()
