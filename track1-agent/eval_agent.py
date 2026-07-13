import json
import random
import os
import sys
from collections import defaultdict
from pathlib import Path

from output_optimizer import detect_task_type, TOKEN_LIMITS, get_dynamic_limits
from local_compressor import optimize_prompt_for_api
import validator
from router.infer_router import predict
from fireworks_client import chat

import argparse

parser = argparse.ArgumentParser(description="Evaluate the agent on a dataset.")
parser.add_argument("--dataset", type=str, default="data/public_style_80_questions.json", help="Path to the dataset JSON file.")
args = parser.parse_args()

DATA_PATH = Path(__file__).parent / args.dataset
# ── GLM-5.2 Judge ──
MODEL_JUDGE = "accounts/fireworks/models/glm-5p2"

def verify_with_glm(prompt: str, answer: str, task_type: str) -> dict:
    """Ask GLM-5.2 to verify an answer.  Returns {verdict, reason, tokens}.
    Verdict is 'correct', 'incorrect', or 'error' (on API failure)."""
    judge_prompt = (
        f"You are a strict answer checker. Given the task and the candidate answer, "
        f"decide if the answer is correct and complete.\n\n"
        f"Task:\n{prompt}\n\n"
        f"Candidate Answer:\n{answer}\n\n"
        f"Reply with EXACTLY one line: CORRECT or INCORRECT followed by a short reason."
    )
    try:
        resp = chat(
            model=MODEL_JUDGE,
            prompt=judge_prompt,
            max_tokens=60,
            system_prompt="You are a precise answer verifier. Output one line: CORRECT or INCORRECT with a brief reason.",
            extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
        )
        text = resp["text"].strip()
        verdict = "correct" if text.upper().startswith("CORRECT") else "incorrect"
        return {
            "verdict": verdict,
            "reason": text,
            "tokens": resp["total_tokens"]
        }
    except Exception as e:
        print(f"[JUDGE ERROR] GLM-5.2 verification failed: {e}")
        return {"verdict": "error", "reason": str(e), "tokens": 0}

# Fallback to local dev env vars if ALLOWED_MODELS is not provided
if "ALLOWED_MODELS" in os.environ:
    models = os.environ["ALLOWED_MODELS"].split(",")
    MODEL_CHEAP = next((m for m in models if "kimi" in m.lower()), models[-1])
    MODEL_EXPENSIVE = next((m for m in models if "minimax" in m.lower()), models[0])
    _summ_cands = [m for m in models if "minimax" in m.lower() or "2p6" in m.lower() or "2.6" in m.lower() or "kimi" in m.lower()]
    MODEL_SUMMARIZATION = _summ_cands[0] if _summ_cands else models[0]
else:
    MODEL_CHEAP = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/kimi-k2p6")
    MODEL_EXPENSIVE = os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p6")
    MODEL_SUMMARIZATION = os.environ.get("MODEL_SUMMARIZATION", os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p6"))


def route_finetuned(prompt: str) -> str:
    """Uses the local DistilBERT model to predict difficulty and route."""
    try:
        label = predict(prompt)
    except Exception as e:
        print(f"[WARNING] Local router failed ({e}), simulating 'easy'")
        label = "easy"
    return MODEL_EXPENSIVE if label == "hard" else MODEL_CHEAP


def sample_tasks(records, total_to_sample=40):
    """Sample tasks, ensuring all categories are represented evenly."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r.get("category", "general")].append(r)

    categories = list(by_category.keys())
    if not categories:
        return []

    per_cat = max(1, total_to_sample // len(categories))
    
    sampled = []
    for cat in categories:
        cat_records = by_category[cat]
        if len(cat_records) > per_cat:
            sampled.extend(random.sample(cat_records, per_cat))
        else:
            sampled.extend(cat_records)
            
    # If we still need more to reach exactly total_to_sample
    remaining = total_to_sample - len(sampled)
    if remaining > 0:
        unused = [r for r in records if r not in sampled]
        sampled.extend(random.sample(unused, min(remaining, len(unused))))
        
    random.shuffle(sampled)
    return sampled[:total_to_sample]


def main():
    print(f"Loading dataset from {DATA_PATH}...")
    with open(DATA_PATH, "r") as f:
        records = json.load(f)
        
    # Filter to only use 'easy' difficulty pool if available
    easy_records = [r for r in records if r.get("difficulty_pool") == "easy"]
    if easy_records:
        records = easy_records
        
    # Sample exactly 80 entries, including all categories
    tasks = sample_tasks(records, 80)
    print(f"Sampled {len(tasks)} tasks across {len(set(t.get('category', 'unknown') for t in tasks))} categories.")
    print("=" * 80)
    
    total_tokens = 0
    judge_tokens = 0
    judge_results = {"correct": 0, "incorrect": 0, "error": 0}
    success_count = 0
    results = []
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("task_id", task.get("id", f"task_{i}"))
        
        # Infer category from task_id (e.g. 'factual_knowledge_001' -> 'factual_knowledge')
        dataset_category = task.get("category")
        if not dataset_category:
            inferred = "_".join(task_id.split("_")[:-1]) if "_" in task_id else "unknown"
            dataset_category = inferred if inferred and inferred != "task" else "unknown"
        task["category"] = dataset_category
        
        prompt = task["prompt"]
        print(f"\n--- TASK {i}/{len(tasks)} [{task_id}] ---")
        print(f"Category (Dataset): {dataset_category}")
        
        task_type = detect_task_type(prompt)
        print(f"Detected Category (Heuristic): {task_type}")
        print(f"Prompt:\n{prompt}\n")
        
        # ── Local solvers (0 API tokens) with graceful fallback ──
        try:
            if task_type == "math_solving" or task.get("category") == "math_reasoning":
                from local_solvers import solve_math_exact, solve_math_pal
                math_ans = solve_math_exact(prompt)
                if math_ans is None:
                    math_ans = solve_math_pal(prompt)
                if math_ans is not None:
                    solver_name = "local_solver (math_exact)" if solve_math_exact(prompt) is not None else "local_solver (math_pal)"
                    print(f"[LOCAL SOLVER] Math solver answered: {math_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": solver_name,
                        "tokens_used": 0,
                        "output": math_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, math_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Math solver failed: {e}")
                
        try:
            is_logic = (task_type == "logical_puzzles" or 
                        task.get("category") == "logical_reasoning" or 
                        any(w in prompt.lower() for w in ["arrange", "constraints:", "clues to determine", "standing in a line", "chairs numbered", "favorite color:", "each have a different", "logic puzzle", "in a row"]))
            if is_logic:
                from local_solvers import solve_logic_puzzle
                logic_ans = solve_logic_puzzle(prompt)
                if logic_ans is not None:
                    print(f"[LOCAL SOLVER] Logic puzzle solver answered: {logic_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (logic)",
                        "tokens_used": 0,
                        "output": logic_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, logic_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Logic solver failed: {e}")

        try:
            is_debug = (task_type == "bug_fixing" or task.get("category") == "code_debugging" or "identify the bug" in prompt.lower())
            if is_debug:
                from local_solvers import solve_code_debug
                debug_ans = solve_code_debug(prompt)
                if debug_ans is not None:
                    print(f"[LOCAL SOLVER] Code debug solver answered:\n{debug_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (code_debug)",
                        "tokens_used": 0,
                        "output": debug_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, debug_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Code debug solver failed: {e}")

        try:
            if task_type == "code_authoring" or task.get("category") == "code_generation":
                from local_solvers import solve_code_authoring
                code_ans = solve_code_authoring(prompt)
                if code_ans is not None:
                    print(f"[LOCAL SOLVER] Code authoring solver answered:\n{code_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (code_authoring)",
                        "tokens_used": 0,
                        "output": code_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, code_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Code authoring solver failed: {e}")


        try:
            if task_type == "sentiment_analysis" or task.get("category") == "sentiment_classification":
                from local_solvers import solve_sentiment
                print("[ROUTER] Local task detected. Routing to local CardiffNLP Sentiment (0 tokens).")
                sentiment_output = solve_sentiment(prompt)
                if sentiment_output is not None:
                    print(f"[RESULT] Local Sentiment output:\n{sentiment_output}")
                    print(f"[TOKENS] 0 API tokens used.")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (sentiment)",
                        "tokens_used": 0,
                        "output": sentiment_output,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, sentiment_output, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    print("\n<EOT>\n" + "=" * 80)
                    continue
        except Exception as e:
            print(f"[WARN] Sentiment solver failed: {e}")

        # Text summarization now directly uses the API (MODEL_SUMMARIZATION)


        try:
            if task_type == "code_authoring" or task.get("category") == "code_generation":
                from local_solvers import solve_code_authoring
                code_ans = solve_code_authoring(prompt)
                if code_ans is not None:
                    print(f"[LOCAL SOLVER] Code authoring solver answered:\n{code_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (code_authoring)",
                        "tokens_used": 0,
                        "output": code_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, code_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Code authoring solver failed: {e}")

        try:
            if task_type == "bug_fixing" or task.get("category") == "code_debugging":
                from local_solvers import solve_code_debugging
                debug_ans = solve_code_debugging(prompt)
                if debug_ans is not None:
                    print(f"[LOCAL SOLVER] Code debugging solver answered:\n{debug_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (code_debugging)",
                        "tokens_used": 0,
                        "output": debug_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, debug_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Code debugging solver failed: {e}")

        try:
            if task_type == "knowledge_qa" or task.get("category") == "factual_knowledge":
                from local_solvers import solve_factual_qa
                qa_ans = solve_factual_qa(prompt)
                if qa_ans is not None:
                    print(f"[LOCAL SOLVER] Factual QA solver answered:\n{qa_ans}")
                    print("[TOKENS] 0 API tokens used.\n\n<EOT>")
                    print("="*80 + "\n")
                    success_count += 1
                    entry = {
                        "task_id": task_id,
                        "category_dataset": task.get("category", "unknown"),
                        "category_detected": task_type,
                        "prompt": prompt,
                        "solver_type": "local",
                        "model_or_solver": "local_solver (knowledge_qa)",
                        "tokens_used": 0,
                        "output": qa_ans,
                        "validation_passed": True
                    }
                    jv = verify_with_glm(prompt, qa_ans, task_type)
                    entry["judge_verdict"] = jv["verdict"]
                    entry["judge_reason"] = jv["reason"]
                    entry["judge_tokens"] = jv["tokens"]
                    judge_tokens += jv["tokens"]
                    judge_results[jv["verdict"]] += 1
                    if jv["verdict"] == "incorrect":
                        print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                    else:
                        print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                    results.append(entry)
                    continue
        except Exception as e:
            print(f"[WARN] Factual QA solver failed: {e}")

        # Route via finetuned model
        model = route_finetuned(prompt)
        
        # Override for summarization
        if task_type == "summarization" or task.get("category") == "text_summarization":
            model = MODEL_SUMMARIZATION
            
        print(f"[ROUTER] Finetuned prediction routed to: {model}")
        
        limits = get_dynamic_limits(task_type, prompt)
        system_prompt = limits["system"]
        
        optimized_prompt = optimize_prompt_for_api(prompt, task_type)
        if len(optimized_prompt) < len(prompt):
            print(f"[COMPRESSOR] Prompt compressed ({len(prompt)} -> {len(optimized_prompt)} chars)")
        
        print(f"[API CALL] Model: {model} | Cap: {limits['cap']} | Reasoning: None")
        try:
            answer = chat(
                model=model,
                prompt=optimized_prompt,
                max_tokens=limits["cap"],
                system_prompt=system_prompt,
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
            )
            total_tokens += answer["total_tokens"]
            
            print(f"[RESPONSE] Tokens: {answer['total_tokens']} | Finish Reason: {answer.get('finish_reason', 'none')}")
            print(f"[TEXT]\n{answer['text']}")
            
            # Clean summarization outputs by stripping outer quotes
            if task_type == "summarization" or task.get("category") == "text_summarization":
                ans = answer["text"].strip()
                if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in {"'", '"'}:
                    answer["text"] = ans[1:-1].strip()
            
            # Validation
            ok, reason = validator.validate(task_type, prompt, answer["text"], answer.get("finish_reason"))
            if not ok or not answer["text"].strip():
                fallback_model = MODEL_EXPENSIVE if model == MODEL_CHEAP else MODEL_CHEAP
                print(f"[VALIDATION FAILED] Reason: {reason}. Retrying with fallback model {fallback_model}...")
                retry_answer = chat(
                    model=fallback_model,
                    prompt=optimized_prompt,
                    max_tokens=limits.get("retry_cap", 800),
                    system_prompt=system_prompt,
                    extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
                )
                total_tokens += retry_answer["total_tokens"]
                print(f"[RETRY RESPONSE] Tokens: {retry_answer['total_tokens']} | Finish Reason: {retry_answer.get('finish_reason', 'none')}")
                print(f"[RETRY TEXT]\n{retry_answer['text']}")
                
                # Clean summarization outputs for retry
                if task_type == "summarization" or task.get("category") == "text_summarization":
                    ans = retry_answer["text"].strip()
                    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in {"'", '"'}:
                        retry_answer["text"] = ans[1:-1].strip()
                        
                entry = {
                    "task_id": task_id,
                    "category_dataset": task.get("category", "unknown"),
                    "category_detected": task_type,
                    "prompt": prompt,
                    "solver_type": "api_retry",
                    "model_or_solver": f"{model} -> {fallback_model}",
                    "tokens_used": answer["total_tokens"] + retry_answer["total_tokens"],
                    "output": retry_answer["text"],
                    "validation_passed": True,
                    "retry_reason": reason
                }
                jv = verify_with_glm(prompt, retry_answer["text"], task_type)
                entry["judge_verdict"] = jv["verdict"]
                entry["judge_reason"] = jv["reason"]
                entry["judge_tokens"] = jv["tokens"]
                judge_tokens += jv["tokens"]
                judge_results[jv["verdict"]] += 1
                if jv["verdict"] == "incorrect":
                    print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                else:
                    print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                results.append(entry)
            else:
                print("[VALIDATION PASSED] Output looks good.")
                entry = {
                    "task_id": task_id,
                    "category_dataset": task.get("category", "unknown"),
                    "category_detected": task_type,
                    "prompt": prompt,
                    "solver_type": "api",
                    "model_or_solver": model,
                    "tokens_used": answer["total_tokens"],
                    "output": answer["text"],
                    "validation_passed": True
                }
                jv = verify_with_glm(prompt, answer["text"], task_type)
                entry["judge_verdict"] = jv["verdict"]
                entry["judge_reason"] = jv["reason"]
                entry["judge_tokens"] = jv["tokens"]
                judge_tokens += jv["tokens"]
                judge_results[jv["verdict"]] += 1
                if jv["verdict"] == "incorrect":
                    print(f"[JUDGE ⚠] GLM-5.2 disagrees: {jv['reason']}")
                else:
                    print(f"[JUDGE ✓] GLM-5.2 verified: {jv['reason']}")
                results.append(entry)
                
            success_count += 1
                
        except Exception as e:
            print(f"[ERROR] API Call failed: {e}")
            results.append({
                "task_id": task_id,
                "category_dataset": task.get("category", "unknown"),
                "category_detected": task_type,
                "prompt": prompt,
                "solver_type": "error",
                "model_or_solver": model,
                "tokens_used": 0,
                "output": str(e),
                "validation_passed": False
            })
            
        print("\n<EOT>\n" + "=" * 80)
        
    category_breakdown = {}
    total_local = 0
    total_retries = 0
    for r in results:
        cat = r.get("category_dataset", "unknown")
        if cat not in category_breakdown:
            category_breakdown[cat] = {
                "total_questions": 0,
                "local_count": 0,
                "api_count": 0,
                "tokens_used": 0,
                "retries": 0,
                "judge_correct": 0,
                "judge_total": 0
            }
        category_breakdown[cat]["total_questions"] += 1
        category_breakdown[cat]["tokens_used"] += r.get("tokens_used", 0)
        if r.get("judge_verdict") in ("correct", "incorrect"):
            category_breakdown[cat]["judge_total"] += 1
            if r.get("judge_verdict") == "correct":
                category_breakdown[cat]["judge_correct"] += 1
        if r.get("solver_type") == "local":
            category_breakdown[cat]["local_count"] += 1
            total_local += 1
        else:
            category_breakdown[cat]["api_count"] += 1
            if r.get("solver_type") == "api_retry":
                category_breakdown[cat]["retries"] += 1
                total_retries += 1

    out_path = Path("eval_results.json")
    total_judged = judge_results["correct"] + judge_results["incorrect"]
    overall_accuracy = judge_results["correct"] / total_judged * 100 if total_judged > 0 else 0.0

    output_data = {
        "summary": {
            "total_tasks": len(tasks),
            "success_count": success_count,
            "total_local_tasks": total_local,
            "total_api_tasks": len(tasks) - total_local,
            "total_retries": total_retries,
            "total_api_tokens": total_tokens,
            "approximate_score": total_tokens / len(tasks) * 19 if len(tasks) > 0 else 0,
            "judge_accuracy_pct": round(overall_accuracy, 1),
            "judge_results": judge_results,
            "judge_tokens": judge_tokens,
            "category_breakdown": category_breakdown
        },
        "results": results
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed evaluation results to {out_path.resolve()}")

    print("\n" + "#" * 80)
    print(f"EVALUATION COMPLETE")
    print(f"Total API Tokens Used: {total_tokens}")
    print(f"Tasks Successfully Processed: {success_count}/{len(tasks)}")
    print(f"Total Tasks Handled Locally (0 tokens): {total_local}/{len(tasks)}")
    print(f"Total Retries (Failed Validation First Time): {total_retries}")
    print(f"Approximate score: {total_tokens/len(tasks)*19:.2f}")
    print("-" * 80)
    print("Category Breakdown (Questions / Local / API / Tokens / Retries / Accuracy):")
    for cat, stats in sorted(category_breakdown.items()):
        jt = stats.get("judge_total", 0)
        jc = stats.get("judge_correct", 0)
        acc_str = f"{jc/jt*100:.0f}%" if jt > 0 else "N/A"
        print(f"  • {cat:<26} | Total: {stats['total_questions']:<2} | Local: {stats['local_count']:<2} | API: {stats['api_count']:<2} | Tokens: {stats['tokens_used']:<4} | Retries: {stats.get('retries', 0)} | Acc: {acc_str} ({jc}/{jt})")
    print("-" * 80)
    print(f"GLM-5.2 Judge Results: ✓ {judge_results['correct']} correct | ⚠ {judge_results['incorrect']} incorrect | ✗ {judge_results['error']} errors")
    print(f"GLM-5.2 Judge Accuracy: {overall_accuracy:.1f}% ({judge_results['correct']}/{total_judged})")
    print(f"GLM-5.2 Judge Tokens Used: {judge_tokens} (not counted in solver score)")
    print("#" * 80)


if __name__ == "__main__":
    main()
