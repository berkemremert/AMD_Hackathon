"""Orchestrator: load tasks → classify → route → call Fireworks → write results.

Guarantees: every task_id from the input appears in the output with a string
answer, valid JSON is always written, and the process exits 0 even if
individual tasks fail.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import classifier
import fireworks_client
import router
import validator

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

RETRY_MODEL = "kimi-k2p7"
# Retry runs with thinking ON, so the cap must be generous: reasoning is
# billed inside completion_tokens, and a tight cap lets it eat the whole
# budget and return empty content (the judge bug, relived).
RETRY_MAX_TOKENS = 4096


def _call_chain(models, prompt, instruction, max_tokens, extra_params):
    """Try models in order, skipping any that are undeployed here."""
    last_err = None
    for model in models:
        try:
            text, usage = fireworks_client.chat_with_usage(
                model=model, prompt=prompt, instruction=instruction,
                max_tokens=max_tokens, extra_params=extra_params,
            )
            return model, text, usage
        except fireworks_client.ModelUnavailableError as e:
            last_err = e
            print(f"[fallback] {model.split('#')[0].rsplit('/', 1)[-1]} unavailable", file=sys.stderr, flush=True)
    raise last_err or RuntimeError("no models available")


def _sum_usage(usages):
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for u in usages:
        for k in out:
            out[k] += u.get(k, 0) or 0
    return out


def answer_task(task):
    """Full pipeline for one task: classify → route → call → validate →
    (maybe) one thinking-ON retry. Never raises."""
    prompt = task.get("prompt", "") or ""
    rec = {"predicted_category": "general", "answer": "", "model": "",
           "usage": {}, "retried": False, "validation": None, "error": None}
    usages = []
    try:
        category = classifier.classify(prompt)
        rec["predicted_category"] = category
        cfg = router.route(category)
        model, text, usage = _call_chain(
            [cfg["model"]] + cfg["fallback_models"],
            prompt, cfg["instruction"], cfg["max_tokens"], cfg["extra_params"],
        )
        usages.append(usage)
        rec["model"], rec["answer"] = model, text

        ok, reason = validator.validate(category, prompt, text, usage.get("finish_reason"))
        if not ok:
            rec["retried"], rec["validation"] = True, reason
            retry_primary = router.resolve_model(RETRY_MODEL)
            retry_chain = [retry_primary] + [m for m in router._allowed_models() if m != retry_primary]
            try:
                m2, text2, usage2 = _call_chain(
                    retry_chain, prompt, cfg["instruction"], RETRY_MAX_TOKENS,
                    extra_params={},  # thinking ON
                )
                usages.append(usage2)
                ok2, _ = validator.validate(category, prompt, text2, usage2.get("finish_reason"))
                # prefer the retry if it validates, or if it at least has
                # content where the original had none
                if ok2 or (text2.strip() and not text.strip()):
                    rec["model"], rec["answer"] = m2, text2
            except Exception as e2:
                if not text.strip():
                    rec["error"] = f"retry failed: {e2}"
    except Exception as e:
        rec["error"] = str(e)
    rec["usage"] = _sum_usage(usages)
    return rec


def run_task(task):
    task_id = task.get("task_id", "")
    rec = answer_task(task)
    if rec["error"]:
        print(f"[fail] {task_id}: {rec['error']}", file=sys.stderr, flush=True)
    else:
        note = f" retried({rec['validation']})" if rec["retried"] else ""
        model = rec["model"].split("#")[0].rsplit("/", 1)[-1] if rec["model"] else "-"
        print(f"[ok] {task_id} category={rec['predicted_category']} model={model}{note}", flush=True)
    return task_id, rec["answer"]


def main():
    """Returns the process exit code: 0 on success, 1 on true failure
    (unreadable input / unwritable output)."""
    fireworks_client.load_dotenv_fallback()

    fatal = False
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[fatal] cannot read {INPUT_PATH}: {e}", file=sys.stderr, flush=True)
        tasks = []
        fatal = True
    if not isinstance(tasks, list):
        print(f"[fatal] {INPUT_PATH} is not a JSON list", file=sys.stderr, flush=True)
        tasks = []
        fatal = True

    answers = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(run_task, t) for t in tasks if isinstance(t, dict)]
            for fut in as_completed(futures):
                task_id, answer = fut.result()
                answers[task_id] = answer

    # Final guard: emit in input order; every task_id present with a
    # non-empty string answer no matter what failed upstream.
    results = [
        {
            "task_id": t.get("task_id", ""),
            "answer": (answers.get(t.get("task_id", "")) or "").strip() or "N/A",
        }
        for t in tasks
        if isinstance(t, dict)
    ]

    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[fatal] cannot write {OUTPUT_PATH}: {e}", file=sys.stderr, flush=True)
        return 1
    print(f"[done] wrote {len(results)} results to {OUTPUT_PATH}", flush=True)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
