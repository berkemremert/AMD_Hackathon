"""Compares 4 routing policies on the same held-out test split used to
evaluate the fine-tuned router (see router/train_router.py's stratified_split,
same random seed, so this is the exact same 17 examples, not a different cut).

  always-cheap     - every query goes to MODEL_CHEAP (real recorded tokens/correctness)
  always-expensive - every query goes to MODEL_EXPENSIVE (real recorded tokens/correctness)
  baseline         - prompt-based classifier decides per query (real Fireworks call,
                      extra tokens every time), then routes to cheap or expensive
  finetuned        - local DistilBERT router decides per query (zero extra tokens),
                      then routes to cheap or expensive

For baseline/finetuned, the actual answer is never re-generated - since the router's
decision determines which of the already-recorded cheap/expensive results would have
been used, we look up the real recorded tokens/correctness for that choice instead of
re-calling Fireworks a second time.
"""
import json
from pathlib import Path

from router.train_router import stratified_split
from router.infer_router import predict as finetuned_predict
from src.baseline_router import classify as baseline_classify

DATA_PATH = Path(__file__).parent / "data" / "labeled_dataset.json"


def eval_always(records, key):
    tokens = sum(r[f"{key}_tokens"] for r in records)
    correct = sum(1 for r in records if r[f"{key}_correct"])
    return {"total_tokens": tokens, "accuracy": correct / len(records)}


def eval_routed(records, route_fn, routing_cost_fn):
    tokens = 0
    correct = 0
    for r in records:
        label = route_fn(r["prompt"])
        routing_tokens = routing_cost_fn(r)
        if label == "hard":
            tokens += routing_tokens + r["expensive_tokens"]
            correct += 1 if r["expensive_correct"] else 0
        else:
            tokens += routing_tokens + r["cheap_tokens"]
            correct += 1 if r["cheap_correct"] else 0
    return {"total_tokens": tokens, "accuracy": correct / len(records)}


def main():
    records = json.loads(DATA_PATH.read_text())
    _, test_records = stratified_split(records)
    n_hard = sum(1 for r in test_records if r["label"] == "hard")
    print(f"Evaluating on {len(test_records)} held-out queries ({n_hard} genuinely hard)\n")

    results = {}
    results["always-cheap"] = eval_always(test_records, "cheap")
    results["always-expensive"] = eval_always(test_records, "expensive")

    baseline_tokens_cache = {}

    def baseline_route_and_cache(prompt):
        result = baseline_classify(prompt)
        baseline_tokens_cache[prompt] = result["tokens"]
        return result["label"]

    results["baseline"] = eval_routed(
        test_records, baseline_route_and_cache, lambda r: baseline_tokens_cache.get(r["prompt"], 0)
    )
    results["finetuned"] = eval_routed(test_records, finetuned_predict, lambda r: 0)

    print(f"{'Approach':<18} {'Total tokens':<14} {'Accuracy':<10}")
    print("-" * 44)
    for name, m in results.items():
        print(f"{name:<18} {m['total_tokens']:<14} {m['accuracy']:.1%}")

    Path(__file__).parent.joinpath("evaluation_results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote evaluation_results.json")


if __name__ == "__main__":
    main()