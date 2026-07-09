"""
Evaluate — sweep the router's confidence threshold to find the sweet spot.

For each threshold we run the router over all tasks many times (so the mock
models' randomness averages out) and report:
  - average accuracy
  - average number of paid (remote) calls
  - average remote tokens (the cost)

Then we pick the SWEET SPOT: the lowest-cost threshold whose average accuracy
still clears the accuracy line. Staying just above the line for the fewest
tokens is exactly how the leaderboard is won.

Run from the project root:
    python src/evaluate.py

AT KICKOFF: set TARGET_ACCURACY to the real accuracy line the hackathon gives
you, and adjust THRESHOLDS / SAMPLES / TRIALS as you like.
"""

import json
import os

from tokens import TokenLedger
from agent import run_agent
from scoring import is_correct

HERE = os.path.dirname(__file__)
TASKS_PATH = os.path.join(HERE, "..", "tasks", "fake_tasks.json")

THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SAMPLES = 5
TRIALS = 30               # repeat each setting this many times to average out luck
TARGET_ACCURACY = 0.90    # the accuracy line we must stay above (set to the real one at kickoff)


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_trial(tasks, threshold):
    """Run the router over all tasks once. Returns (accuracy, remote_calls, remote_tokens)."""
    ledger = TokenLedger()
    correct = 0
    for task in tasks:
        answer = run_agent(task, ledger, mode="router", threshold=threshold, samples=SAMPLES)
        if is_correct(answer, task["answer"]):
            correct += 1
    remote_calls = sum(1 for c in ledger.calls if c["where"] == "remote")
    return correct / len(tasks), remote_calls, ledger.remote_total


def evaluate_threshold(tasks, threshold):
    """Average many trials at one threshold. Returns (avg_acc, avg_calls, avg_tokens)."""
    acc_sum = calls_sum = tok_sum = 0.0
    for _ in range(TRIALS):
        acc, calls, toks = one_trial(tasks, threshold)
        acc_sum += acc
        calls_sum += calls
        tok_sum += toks
    return acc_sum / TRIALS, calls_sum / TRIALS, tok_sum / TRIALS


def main():
    tasks = load_tasks(TASKS_PATH)

    print(f"Sweeping thresholds, {TRIALS} trials each "
          f"(accuracy line = {int(TARGET_ACCURACY * 100)}%)\n")
    header = (f"{'threshold':<12}{'avg accuracy':<16}"
              f"{'avg remote calls':<20}{'avg remote tokens':<18}")
    print(header)
    print("-" * len(header))

    results = []
    for t in THRESHOLDS:
        acc, calls, toks = evaluate_threshold(tasks, t)
        results.append((t, acc, calls, toks))
        acc_str = f"{acc * 100:.0f}%"
        calls_str = f"{calls:.1f}"
        toks_str = f"{toks:.1f}"
        flag = "  <- clears the line" if acc >= TARGET_ACCURACY else ""
        print(f"{t:<12}{acc_str:<16}{calls_str:<20}{toks_str:<18}{flag}")

    # Sweet spot: among thresholds that clear the line, the one with the fewest tokens.
    passing = [r for r in results if r[1] >= TARGET_ACCURACY]
    print("\n" + "=" * len(header))
    if passing:
        best = min(passing, key=lambda r: r[3])   # fewest tokens among passers
        print(f"SWEET SPOT: threshold = {best[0]}  ->  "
              f"{best[1] * 100:.0f}% accuracy at ~{best[3]:.0f} remote tokens.")
        print("Lowest-cost setting that still clears the accuracy line. "
              "Notice it's far cheaper than sending everything to remote.")
    else:
        print("No threshold cleared the line with the current mock local model.")
        print("At kickoff the real models apply; you can also raise the threshold,")
        print("improve the local model, or increase SAMPLES.")


if __name__ == "__main__":
    main()
