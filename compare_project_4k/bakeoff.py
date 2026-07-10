#!/usr/bin/env python3
"""Dev-only model bakeoff. NOT part of the submission pipeline.

Runs every task in tests/tasks.json through EACH allowed model with identical
minimal prompting (no system prompt, temperature=0), judges answers with an
LLM judge, and reports per-model/per-category accuracy and token costs.

Also probes reasoning/thinking behavior:
  - inspects raw responses for reasoning fields / <think> tags
  - measures completion_tokens on a trivial task (100+ = hidden reasoning)
  - for minimax-m3 and kimi-k2p7-code, tries several ways to disable thinking
    and reports on/off token counts.

Usage:
    python3 bakeoff.py                # full 40 x 5 run + probes
    python3 bakeoff.py --sample 8     # stratified subset per model
    python3 bakeoff.py --probes-only  # skip the 40x5 matrix
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

import fireworks_client
import router
from eval import JUDGE_SYSTEM, JUDGE_TEMPLATE, stratified_sample

import os

WORKERS = 8
GEN_MAX_TOKENS = 3072  # generous so reasoning models aren't truncated mid-think
TRIVIAL_PROMPT = "Classify sentiment: 'I love this.' One word only."
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
CAT_ORDER = ["factual", "math", "sentiment", "summarization", "ner", "code_debug", "logic", "code_gen"]
CAT_SHORT = {"factual": "fact", "math": "math", "sentiment": "sent", "summarization": "summ",
             "ner": "ner", "code_debug": "debug", "logic": "logic", "code_gen": "gen"}


def raw_chat(model, prompt, max_tokens=GEN_MAX_TOKENS, extra=None, retries=3):
    """Direct call returning the FULL response dict so we can inspect it."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    headers = {
        "Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
        "Content-Type": "application/json",
    }
    url = os.environ["FIREWORKS_BASE_URL"].rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(2 ** (attempt - 1))
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
        except requests.RequestException as e:
            last = str(e)
            continue
        if resp.status_code in (408, 429, 500, 502, 503, 504):
            last = f"HTTP {resp.status_code}"
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()
    raise RuntimeError(f"exhausted retries: {last}")


def visible_answer(message):
    """Strip <think> blocks; if an unclosed <think> ate the whole output, return ''."""
    content = message.get("content") or ""
    content = THINK_RE.sub("", content)
    if "<think>" in content:  # unclosed think block (truncated)
        content = content.split("<think>")[0]
    return content.strip()


def reasoning_info(data):
    """What reasoning artifacts does this response carry?"""
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    extra_fields = {
        k: len(str(v)) for k, v in msg.items()
        if k not in ("role", "content", "tool_calls", "function_call") and v
    }
    return {
        "reasoning_fields": extra_fields,            # e.g. {"reasoning_content": 1234}
        "think_tags_in_content": "<think>" in content,
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def short(model_id):
    # "accounts/.../models/gemma-4-31b-it#accounts/.../deployments/xyz" -> "gemma-4-31b-it"
    return model_id.split("#")[0].rsplit("/", 1)[-1]


# ---------------- probes ----------------

def probe_trivial(models):
    print("\n" + "=" * 76)
    print("PROBE 1: trivial task per model — completion_tokens should be tiny")
    print(f'  prompt: "{TRIVIAL_PROMPT}"')
    print(f"  {'model':<24}{'compl_toks':>11}  {'answer':<12} reasoning artifacts")
    results = {}
    for m in models:
        try:
            data = raw_chat(m, TRIVIAL_PROMPT, max_tokens=2048)
            usage = data.get("usage", {})
            info = reasoning_info(data)
            ans = visible_answer(data["choices"][0]["message"])[:12]
            arts = []
            if info["reasoning_fields"]:
                arts.append("fields=" + ",".join(f"{k}({v} chars)" for k, v in info["reasoning_fields"].items()))
            if info["think_tags_in_content"]:
                arts.append("<think> in content")
            ct = usage.get("completion_tokens", -1)
            flag = "  <-- HIDDEN REASONING" if ct >= 100 else ""
            print(f"  {short(m):<24}{ct:>11}  {ans:<12} {'; '.join(arts) or 'none'}{flag}")
            results[m] = {"completion_tokens": ct, **info}
        except Exception as e:
            print(f"  {short(m):<24}{'ERR':>11}  {str(e)[:50]}")
            results[m] = {"error": str(e)}
    return results


def probe_thinking_toggle(models):
    """Try documented/likely ways to disable thinking on reasoning models."""
    variants = [
        ("baseline", {}, None),
        ("reasoning_effort=none", {"reasoning_effort": "none"}, None),
        ("reasoning_effort=low", {"reasoning_effort": "low"}, None),
        ("thinking.type=disabled", {"thinking": {"type": "disabled"}}, None),
        ("chat_template_kwargs.enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}, None),
        ("prompt /no_think prefix", {}, "/no_think\n"),
    ]
    out = {}
    print("\n" + "=" * 76)
    print("PROBE 2: can thinking be disabled? (trivial task, per variant)")
    for m in models:
        print(f"\n  {short(m)}:")
        out[m] = {}
        for name, extra, prefix in variants:
            prompt = (prefix or "") + TRIVIAL_PROMPT
            try:
                data = raw_chat(m, prompt, max_tokens=2048, extra=extra, retries=1)
                usage = data.get("usage", {})
                info = reasoning_info(data)
                ct = usage.get("completion_tokens", -1)
                ans = visible_answer(data["choices"][0]["message"])[:20].replace("\n", " ")
                arts = "think-tags" if info["think_tags_in_content"] else (
                    ",".join(info["reasoning_fields"]) or "no-reasoning")
                print(f"    {name:<44}{ct:>6} compl_toks  [{arts}]  ans={ans!r}")
                out[m][name] = {"completion_tokens": ct, "artifacts": arts, "answer": ans}
            except Exception as e:
                print(f"    {name:<44}   ERR  {str(e)[:70]}")
                out[m][name] = {"error": str(e)[:200]}
    return out


def accessible(model):
    """1-token preflight: is this model reachable with the current key?"""
    try:
        raw_chat(model, "hi", max_tokens=1, retries=0)
        return True
    except Exception:
        return False


def pick_judge():
    """Prefer the gemma judge; fall back to kimi. Always disable thinking —
    every model here reasons by default, and with a tiny judge max_tokens the
    reasoning would eat the whole budget and leave content empty."""
    gemma = router.resolve_model("gemma-4-31b-it")
    if accessible(gemma):
        return gemma, {"reasoning_effort": "none"}
    kimi = router.resolve_model("kimi-k2p7")
    return kimi, {"reasoning_effort": "none"}


# ---------------- full matrix ----------------

def run_matrix(models, tasks):
    jobs = [(m, t) for m in models for t in tasks]
    results = {}

    def gen(job):
        m, t = job
        try:
            data = raw_chat(m, t["prompt"])
            msg = data["choices"][0]["message"]
            return (m, t["task_id"]), {
                "answer": visible_answer(msg),
                "usage": data.get("usage", {}),
                "reasoning": reasoning_info(data),
                "error": None,
            }
        except Exception as e:
            return (m, t["task_id"]), {"answer": "", "usage": {}, "reasoning": {}, "error": str(e)}

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for key, rec in pool.map(gen, jobs):
            results[key] = rec
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(jobs)} generations", flush=True)

    judge_model, judge_extra = pick_judge()
    print(f"  judging {len(jobs)} answers with {short(judge_model)}...", flush=True)

    def judge(job):
        m, t = job
        rec = results[(m, t["task_id"])]
        if not rec["answer"]:
            return (m, t["task_id"]), False
        try:
            extra = dict(judge_extra)
            extra["messages"] = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=t["prompt"], answer=rec["answer"])},
            ]
            data = raw_chat(
                judge_model,
                JUDGE_TEMPLATE.format(prompt=t["prompt"], answer=rec["answer"]),
                max_tokens=4,
                extra=extra,
            )
            verdict = (data["choices"][0]["message"].get("content") or "").strip().upper()
            return (m, t["task_id"]), verdict.startswith("YES")
        except Exception as e:
            print(f"  [judge-error] {t['task_id']}: {e}", file=sys.stderr)
            return (m, t["task_id"]), False

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for key, passed in pool.map(judge, jobs):
            results[key]["passed"] = passed
    return results


def report(models, tasks, results):
    cats = [c for c in CAT_ORDER if any(t["category"] == c for t in tasks)]
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # model -> cat -> [pass, n]
    toks = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0, "n": 0})
    cat_ct = defaultdict(lambda: defaultdict(int))             # model -> cat -> completion toks

    for m in models:
        for t in tasks:
            r = results[(m, t["task_id"])]
            a = acc[m][t["category"]]
            a[0] += bool(r.get("passed"))
            a[1] += 1
            u = r["usage"]
            toks[m]["prompt"] += u.get("prompt_tokens", 0)
            toks[m]["completion"] += u.get("completion_tokens", 0)
            toks[m]["total"] += u.get("total_tokens", 0)
            toks[m]["n"] += 1
            cat_ct[m][t["category"]] += u.get("completion_tokens", 0)

    print("\n" + "=" * 76)
    print("ACCURACY PER CATEGORY (passed/n)")
    hdr = f"  {'model':<24}" + "".join(f"{CAT_SHORT[c]:>7}" for c in cats) + f"{'ALL':>8}"
    print(hdr)
    for m in models:
        row = f"  {short(m):<24}"
        tp = tn = 0
        for c in cats:
            p, n = acc[m][c]
            tp += p
            tn += n
            row += f"{f'{p}/{n}':>7}"
        row += f"{f'{tp}/{tn}':>8}"
        print(row)

    print("\nTOKENS")
    print(f"  {'model':<24}{'avg_compl/task':>15}{'prompt':>10}{'compl':>10}{'TOTAL':>10}")
    for m in models:
        t = toks[m]
        avg = t["completion"] / max(t["n"], 1)
        print(f"  {short(m):<24}{avg:>15.1f}{t['prompt']:>10}{t['completion']:>10}{t['total']:>10}")

    print("\nCOMPLETION TOKENS PER CATEGORY (sum)")
    print(f"  {'model':<24}" + "".join(f"{CAT_SHORT[c]:>7}" for c in cats))
    for m in models:
        print(f"  {short(m):<24}" + "".join(f"{cat_ct[m][c]:>7}" for c in cats))

    print("\nRECOMMENDATION — cheapest model (completion toks) passing each category")
    print("  (pass = >=80% category accuracy; * = nothing passed, best accuracy shown)")
    for c in cats:
        candidates = []
        for m in models:
            p, n = acc[m][c]
            candidates.append((m, p / max(n, 1), cat_ct[m][c]))
        passing = [x for x in candidates if x[1] >= 0.8]
        if passing:
            best = min(passing, key=lambda x: x[2])
            print(f"  {c:<15} -> {short(best[0]):<24} ({best[1]:.0%}, {best[2]} compl toks)")
        else:
            best = max(candidates, key=lambda x: (x[1], -x[2]))
            print(f"  {c:<15} -> {short(best[0]):<24} ({best[1]:.0%}, {best[2]} compl toks) *")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tests/tasks.json")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--probes-only", action="store_true")
    args = ap.parse_args()

    fireworks_client.load_dotenv_fallback()
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    print(f"Models: {', '.join(short(m) for m in models)}")
    print("Preflighting model access...", flush=True)
    live, dead = [], []
    for m in models:
        (live if accessible(m) else dead).append(m)
    if dead:
        print(f"  INACCESSIBLE with this key (skipped): {', '.join(short(m) for m in dead)}")
    models = live
    if not models:
        sys.exit("no accessible models")

    trivial = probe_trivial(models)
    # every allowed model turned out to emit reasoning_content — probe them all
    toggle = probe_thinking_toggle(models)

    matrix_report = None
    if not args.probes_only:
        with open(args.input) as f:
            tasks = json.load(f)
        if args.sample:
            tasks = stratified_sample(tasks, args.sample)
        print("\n" + "=" * 76)
        print(f"FULL MATRIX: {len(tasks)} tasks x {len(models)} models "
              f"(identical minimal prompting, temp=0, max_tokens={GEN_MAX_TOKENS})")
        results = run_matrix(models, tasks)
        report(models, tasks, results)
        matrix_report = {f"{short(m)}|{tid}": v for (m, tid), v in results.items()}

    with open("tests/bakeoff_results.json", "w") as f:
        json.dump({"trivial_probe": {short(k): v for k, v in trivial.items()},
                   "thinking_toggle": {short(k): v for k, v in toggle.items()},
                   "matrix": matrix_report}, f, indent=2, ensure_ascii=False)
    print("\nRaw records written to tests/bakeoff_results.json")


if __name__ == "__main__":
    main()
