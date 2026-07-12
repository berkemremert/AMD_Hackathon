"""Empirically labels each query as "easy" or "hard" based on whether the cheap
model's answer actually passes grading (real Fireworks calls, no mocking):

  - code_generation: graded by executing the generated function against known
    test cases (see ../code_exec.py) — objective pass/fail, no LLM judge needed.
  - everything else: graded by an independent judge model comparing the answer
    against the query's ground_truth/rubric.

label = "easy" if the cheap model's answer passes grading, else "hard".
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fireworks_client import chat
from code_exec import run_tests
import os

import concurrent.futures
import random

IN_PATH = Path(__file__).parent / "queries_raw.json"
OUT_PATH = Path(__file__).parent / "labeled_dataset.json"

MODEL_CHEAP_POOL = os.environ["MODEL_CHEAP"].split(",")
MODEL_EXPENSIVE_POOL = os.environ["MODEL_EXPENSIVE"].split(",")
MODEL_JUDGE = os.environ.get("MODEL_JUDGE", "accounts/fireworks/models/kimi-k2p7-code")

KEYS = []
if "FIREWORKS_API_KEY" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY"])
elif "FIREWORKS_API_KEY_1" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY_1"])

if not KEYS:
    raise ValueError("No Fireworks API keys found. Please set FIREWORKS_API_KEY in .env")

JUDGE_PROMPT = """You are a fast, decisive grader. Do not deliberate at length, do not count words, \
do not second-guess yourself. Grade each candidate in one short sentence, then stop.

Question: {prompt}

Known correct answer or rubric: {ground_truth}

Candidate A: {cheap_answer}

Candidate B: {expensive_answer}

For each candidate, decide if it correctly and adequately answers the question given the \
known correct answer/rubric above. Minor wording differences are fine; factual or logical \
errors are not. Give your verdict immediately - do not show detailed reasoning.

Respond in this exact format and nothing else:
A: <one short sentence> -> <CORRECT or INCORRECT>
B: <one short sentence> -> <CORRECT or INCORRECT>
JSON: {{"a_correct": true or false, "b_correct": true or false}}"""


def parse_judge_json(text: str) -> dict:
    # First try to find the JSON block specifically
    match = re.search(r"\{[^{}]*a_correct[^{}]*\}", text, re.IGNORECASE)
    if match:
        try:
            # Clean up python-style booleans if the model hallucinates them
            clean_str = match.group(0).replace('True', 'true').replace('False', 'false')
            return json.loads(clean_str)
        except json.JSONDecodeError:
            pass

    # Fallback: look for the text verdicts
    a_match = re.search(r"A:.*?(CORRECT|INCORRECT)", text, re.IGNORECASE | re.DOTALL)
    b_match = re.search(r"B:.*?(CORRECT|INCORRECT)", text, re.IGNORECASE | re.DOTALL)
    
    if a_match and b_match:
        a_correct = "INCORRECT" not in a_match.group(1).upper()
        b_correct = "INCORRECT" not in b_match.group(1).upper()
        return {"a_correct": a_correct, "b_correct": b_correct}
        
    raise ValueError(f"Could not parse judge response: {text!r}")


def grade(query: dict, cheap_answer: str, expensive_answer: str, api_key: str = None) -> tuple[bool, bool, int]:
    """Returns (cheap_correct, expensive_correct, judge_tokens_used)."""
    if query["category"] == "code_generation":
        spec = json.loads(query["ground_truth"])
        cheap_ok = run_tests(cheap_answer, spec["function_name"], spec["tests"])
        expensive_ok = run_tests(expensive_answer, spec["function_name"], spec["tests"])
        return cheap_ok, expensive_ok, 0

    prompt = JUDGE_PROMPT.format(
        prompt=query["prompt"],
        ground_truth=query["ground_truth"],
        cheap_answer=cheap_answer,
        expensive_answer=expensive_answer,
    )
    result = chat(MODEL_JUDGE, prompt, max_tokens=4000, temperature=0.0, api_key=api_key)
    verdict = parse_judge_json(result["text"])
    return bool(verdict["a_correct"]), bool(verdict["b_correct"]), result["total_tokens"]


def process_query(args):
    idx, q = args
    api_key = KEYS[idx % len(KEYS)]
    model_cheap = random.choice(MODEL_CHEAP_POOL)
    model_expensive = random.choice(MODEL_EXPENSIVE_POOL)
    
    try:
        cheap = chat(model_cheap, q["prompt"], max_tokens=4000, api_key=api_key)
        expensive = chat(model_expensive, q["prompt"], max_tokens=4000, api_key=api_key)
        cheap_ok, expensive_ok, judge_tokens = grade(q, cheap["text"], expensive["text"], api_key=api_key)
        
        return {
            "id": q["id"],
            "category": q["category"],
            "difficulty_pool": q["difficulty_pool"],
            "prompt": q["prompt"],
            "label": "easy" if cheap_ok else "hard",
            "cheap_model": model_cheap,
            "cheap_correct": cheap_ok,
            "cheap_tokens": cheap["total_tokens"],
            "expensive_model": model_expensive,
            "expensive_correct": expensive_ok,
            "expensive_tokens": expensive["total_tokens"],
            "judge_tokens": judge_tokens,
        }
    except Exception as e:
        print(f"  [{q['id']}] FAILED: {e}", file=sys.stderr)
        return None

def main():
    if not IN_PATH.exists():
        print(f"Error: {IN_PATH} does not exist. Please run generate_dataset.py first.")
        return
        
    queries = json.loads(IN_PATH.read_text())
    labeled = []
    if OUT_PATH.exists():
        try:
            labeled = json.loads(OUT_PATH.read_text())
        except json.JSONDecodeError:
            pass
            
    done_ids = {r["id"] for r in labeled}
    
    tasks = []
    for i, q in enumerate(queries):
        if q["id"] not in done_ids:
            tasks.append((i, q))

    if not tasks:
        easy = sum(1 for r in labeled if r["label"] == "easy")
        hard = len(labeled) - easy
        print(f"Dataset already fully labeled! Total: {len(labeled)} ({easy} easy, {hard} hard).")
        return

    max_workers = len(KEYS) * 10
    print(f"Resuming labeling. Need to label {len(tasks)} more queries. Using {len(KEYS)} API key(s) and {max_workers} workers...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_query, t): t for t in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                t = future_to_task[future]
                result = future.result()
                if result:
                    labeled.append(result)
                    OUT_PATH.write_text(json.dumps(labeled, indent=2))
                    print(f"[{len(labeled)}/{len(queries)}] {result['id']} -> label={result['label']} (cheap={result['cheap_correct']}, exp={result['expensive_correct']})")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved incrementally.")

    easy = sum(1 for r in labeled if r["label"] == "easy")
    hard = len(labeled) - easy
    print(f"\nDone. {len(labeled)} labeled ({easy} easy, {hard} hard). Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
