#!/usr/bin/env python3
"""Dev-only eval harness. NOT part of the submission pipeline.

Runs the full classify→route→call pipeline on tests/tasks.json, then grades
each answer with one LLM-judge call (gemma-4-31b-it). Judge tokens are counted
separately — they never touch the competition score.

Usage:
    python3 eval.py                 # all 40 tasks
    python3 eval.py --sample 16     # stratified subset (2 per category)
"""

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import fireworks_client
import main as pipeline  # 'main' alone would be shadowed by our main() below
import router

JUDGE_MODEL_SUBSTR = "gemma-4-31b-it"
JUDGE_SYSTEM = "You are a strict grader. Reply with only YES or NO."
JUDGE_TEMPLATE = (
    "Task:\n{prompt}\n\nAnswer:\n{answer}\n\n"
    "Does this answer correctly and completely satisfy the task's intent, "
    "including any format/length constraints? Reply YES or NO."
)
WORKERS = 8


def pick_judge_model():
    """Preferred judge, or the first allowed model actually deployed."""
    preferred = router.resolve_model(JUDGE_MODEL_SUBSTR)
    candidates = [preferred] + [m for m in router._allowed_models() if m != preferred]
    for m in candidates:
        try:
            fireworks_client.chat(m, "hi", max_tokens=1, extra_params={"reasoning_effort": "none"})
            return m
        except fireworks_client.ModelUnavailableError:
            continue
        except Exception:
            return m  # reachable but grumpy (e.g. rate limit) — still usable
    sys.exit("no allowed model is deployed; cannot judge")


def stratified_sample(tasks, n):
    """Pick ~n tasks, round-robin across categories to keep coverage."""
    by_cat = OrderedDict()
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t)
    picked, i = [], 0
    while len(picked) < n:
        added = False
        for cat_tasks in by_cat.values():
            if i < len(cat_tasks) and len(picked) < n:
                picked.append(cat_tasks[i])
                added = True
        if not added:
            break
        i += 1
    return picked


def run_pipeline_task(task):
    """Exact production path: main.answer_task (validation + retry included)."""
    rec = pipeline.answer_task(task)
    rec.update(task_id=task["task_id"], category=task["category"], prompt=task["prompt"])
    return rec


def judge_task(rec, judge_model):
    if not rec["answer"]:
        return False, {}
    try:
        # thinking off is mandatory here: a reasoning burst would eat the tiny
        # max_tokens budget and return empty content (= every task judged NO)
        verdict, usage = fireworks_client.chat_with_usage(
            model=judge_model,
            prompt=JUDGE_TEMPLATE.format(prompt=rec["prompt"], answer=rec["answer"]),
            instruction=JUDGE_SYSTEM,
            max_tokens=6,
            extra_params={"reasoning_effort": "none"},
        )
        return verdict.strip().upper().startswith("YES"), usage
    except Exception as e:
        print(f"[judge-error] {rec['task_id']}: {e}", file=sys.stderr)
        return False, {}


def tok(usage, key):
    return usage.get(key, 0) or 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tests/tasks.json")
    ap.add_argument("--sample", type=int, default=0, help="run only N tasks (stratified); 0 = all")
    args = ap.parse_args()

    fireworks_client.load_dotenv_fallback()
    judge_model = pick_judge_model()

    with open(args.input) as f:
        tasks = json.load(f)
    if args.sample:
        tasks = stratified_sample(tasks, args.sample)
    print(f"Running pipeline on {len(tasks)} tasks...", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records = list(pool.map(run_pipeline_task, tasks))

    print(f"Judging {len(records)} answers with {judge_model}...", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        verdicts = list(pool.map(lambda r: judge_task(r, judge_model), records))
    for rec, (passed, judge_usage) in zip(records, verdicts):
        rec["passed"] = passed
        rec["judge_usage"] = judge_usage

    # ---- aggregate ----
    cat_stats = defaultdict(lambda: {"n": 0, "pass": 0, "prompt": 0, "completion": 0, "total": 0})
    model_stats = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    totals = {"prompt": 0, "completion": 0, "total": 0}
    judge_total = 0
    misrouted, failed, retried = [], [], []

    for r in records:
        c = cat_stats[r["category"]]
        c["n"] += 1
        c["pass"] += r["passed"]
        for k, uk in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("total", "total_tokens")):
            c[k] += tok(r["usage"], uk)
            totals[k] += tok(r["usage"], uk)
        if r["model"]:
            m = model_stats[r["model"].rsplit("/", 1)[-1]]
            m["calls"] += 1
            for k, uk in (("prompt", "prompt_tokens"), ("completion", "completion_tokens"), ("total", "total_tokens")):
                m[k] += tok(r["usage"], uk)
        judge_total += tok(r["judge_usage"], "total_tokens")
        if r.get("retried"):
            retried.append(f"{r['task_id']} ({r['validation']})")
        if r["predicted_category"] != r["category"]:
            misrouted.append(f"{r['task_id']} ({r['category']} -> {r['predicted_category']})")
        if not r["passed"]:
            failed.append(r["task_id"] + (f" [error: {r['error']}]" if r["error"] else ""))

    n = len(records)
    npass = sum(r["passed"] for r in records)

    print("\n" + "=" * 68)
    print(f"OVERALL ACCURACY: {npass}/{n} ({100.0 * npass / n:.1f}%)" if n else "no tasks")
    print(f"PIPELINE TOKENS:  prompt={totals['prompt']}  completion={totals['completion']}  TOTAL={totals['total']}")
    print(f"(judge tokens, dev-only, not scored: {judge_total})")

    print("\nPER-CATEGORY:")
    print(f"  {'category':<15}{'acc':>8}{'prompt':>9}{'compl':>8}{'total':>8}")
    for cat in sorted(cat_stats):
        c = cat_stats[cat]
        print(f"  {cat:<15}{c['pass']}/{c['n']:<6}{c['prompt']:>9}{c['completion']:>8}{c['total']:>8}")

    print("\nPER-MODEL:")
    print(f"  {'model':<28}{'calls':>6}{'prompt':>9}{'compl':>8}{'total':>8}")
    for name in sorted(model_stats):
        m = model_stats[name]
        print(f"  {name:<28}{m['calls']:>6}{m['prompt']:>9}{m['completion']:>8}{m['total']:>8}")

    if retried:
        print(f"\nVALIDATION RETRIES ({len(retried)}):")
        for line in retried:
            print(f"  {line}")
    if misrouted:
        print("\nMISCLASSIFIED (intended -> predicted):")
        for line in misrouted:
            print(f"  {line}")
    print("\nFAILED TASKS:" if failed else "\nFAILED TASKS: none")
    for line in failed:
        print(f"  {line}")

    out_path = "tests/eval_results.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed records written to {out_path}")


if __name__ == "__main__":
    main()
