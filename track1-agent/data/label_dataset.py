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

IN_PATH = Path(__file__).parent / "queries_raw.json"
OUT_PATH = Path(__file__).parent / "labeled_dataset.json"

MODEL_CHEAP = os.environ["MODEL_CHEAP"]
MODEL_EXPENSIVE = os.environ["MODEL_EXPENSIVE"]
MODEL_JUDGE = os.environ["MODEL_JUDGE"]

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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(match.group(0))


def grade(query: dict, cheap_answer: str, expensive_answer: str) -> tuple[bool, bool, int]:
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
    result = chat(MODEL_JUDGE, prompt, max_tokens=1200, temperature=0.0)
    verdict = parse_judge_json(result["text"])
    return bool(verdict["a_correct"]), bool(verdict["b_correct"]), result["total_tokens"]


def main():
    queries = json.loads(IN_PATH.read_text())
    labeled = []
    if OUT_PATH.exists():
        labeled = json.loads(OUT_PATH.read_text())
    done_ids = {r["id"] for r in labeled}

    for i, q in enumerate(queries):
        if q["id"] in done_ids:
            continue
        print(f"[{i + 1}/{len(queries)}] {q['id']} ({q['category']})...", flush=True)
        try:
            cheap = chat(MODEL_CHEAP, q["prompt"], max_tokens=700)
            expensive = chat(MODEL_EXPENSIVE, q["prompt"], max_tokens=700)
            cheap_ok, expensive_ok, judge_tokens = grade(q, cheap["text"], expensive["text"])
            record = {
                "id": q["id"],
                "category": q["category"],
                "difficulty_pool": q["difficulty_pool"],
                "prompt": q["prompt"],
                "label": "easy" if cheap_ok else "hard",
                "cheap_correct": cheap_ok,
                "cheap_tokens": cheap["total_tokens"],
                "expensive_correct": expensive_ok,
                "expensive_tokens": expensive["total_tokens"],
                "judge_tokens": judge_tokens,
            }
            labeled.append(record)
            OUT_PATH.write_text(json.dumps(labeled, indent=2))
            print(f"  label={record['label']} cheap_ok={cheap_ok} expensive_ok={expensive_ok}")
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            time.sleep(3)

    easy = sum(1 for r in labeled if r["label"] == "easy")
    hard = len(labeled) - easy
    print(f"\nDone. {len(labeled)} labeled ({easy} easy, {hard} hard). Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
