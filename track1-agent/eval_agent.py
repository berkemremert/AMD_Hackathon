import json
import random
import os
import sys
from collections import defaultdict
from pathlib import Path

from output_optimizer import detect_task_type, TOKEN_LIMITS
from local_solvers import solve_ner
import validator
from router.infer_router import predict
from fireworks_client import chat

DATA_PATH = Path(__file__).parent / "data" / "labeled_dataset.json"

# Fallback to local dev env vars if ALLOWED_MODELS is not provided
if "ALLOWED_MODELS" in os.environ:
    models = os.environ["ALLOWED_MODELS"].split(",")
    # The user wants to use kimi-k2p6 (non-coding version) always
    MODEL_CHEAP = next((m for m in models if "kimi" in m.lower() and "code" not in m.lower()), models[0])
    MODEL_EXPENSIVE = next((m for m in models if "kimi" in m.lower() and "code" not in m.lower()), models[-1])
else:
    MODEL_CHEAP = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/kimi-k2p6")
    MODEL_EXPENSIVE = os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p6")


def route_finetuned(prompt: str) -> str:
    """Uses the local DistilBERT model to predict difficulty and route."""
    label = predict(prompt)
    return MODEL_EXPENSIVE if label == "hard" else MODEL_CHEAP


def sample_tasks(records, total_to_sample=40):
    """Sample tasks, ensuring all categories are represented evenly."""
    by_category = defaultdict(list)
    for r in records:
        by_category[r.get("category", "general")].append(r)
    
    categories = list(by_category.keys())
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
        
    # Filter to only use 'easy' difficulty pool as requested
    records = [r for r in records if r.get("difficulty_pool") == "easy"]
        
    # Sample exactly 40 entries, including all categories
    tasks = sample_tasks(records, 40)
    print(f"Sampled {len(tasks)} tasks across {len(set(t.get('category', 'unknown') for t in tasks))} categories.")
    print("=" * 80)
    
    total_tokens = 0
    success_count = 0
    
    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        prompt = task["prompt"]
        print(f"\n--- TASK {i}/{len(tasks)} [{task_id}] ---")
        print(f"Category (Dataset): {task.get('category', 'unknown')}")
        
        task_type = detect_task_type(prompt)
        print(f"Detected Category (Heuristic): {task_type}")
        print(f"Prompt:\n{prompt}\n")
        
        if task_type == "entity_extraction":
            print("[ROUTER] Local task detected. Routing to local GLiNER (0 tokens).")
            raw_entities = solve_ner(prompt)
            print(f"[RESULT] Local NER output:\n{raw_entities}")
            print(f"[TOKENS] 0 API tokens used.")
            success_count += 1
            print("\n<EOT>\n" + "=" * 80)
            continue
            
        if task_type == "sentiment_analysis":
            from local_solvers import solve_sentiment
            print("[ROUTER] Local task detected. Routing to local CardiffNLP Sentiment (0 tokens).")
            sentiment_output = solve_sentiment(prompt)
            print(f"[RESULT] Local Sentiment output:\n{sentiment_output}")
            print(f"[TOKENS] 0 API tokens used.")
            success_count += 1
            print("\n<EOT>\n" + "=" * 80)
            continue

        # Route via finetuned model
        model = route_finetuned(prompt)
        print(f"[ROUTER] Finetuned prediction routed to: {model}")
        
        limits = TOKEN_LIMITS.get(task_type, TOKEN_LIMITS["fallback"])
        tight_prompt = f"{limits['suffix']}\n\n{prompt}"
        
        print(f"[API CALL] Model: {model} | Cap: {limits['cap']} | Reasoning: Low")
        try:
            answer = chat(model, tight_prompt, max_tokens=limits["cap"], extra_params={"reasoning_effort": "low", "reasoning_history": "disabled"})
            call_tokens = answer["total_tokens"]
            total_tokens += call_tokens
            
            print(f"[RESPONSE] Tokens: {call_tokens} | Finish Reason: {answer.get('finish_reason')}")
            print(f"[TEXT]\n{answer['text']}")
            
            # Validation
            ok, reason = validator.validate(task_type, prompt, answer["text"], answer.get("finish_reason"))
            if not ok:
                print(f"[VALIDATION FAILED] Reason: {reason}. Retrying with thinking OFF and generous cap ({limits.get('retry_cap', 800)})...")
                retry_answer = chat(model, tight_prompt, max_tokens=limits.get("retry_cap", 800), extra_params={"reasoning_effort": "low", "reasoning_history": "disabled"})
                retry_tokens = retry_answer["total_tokens"]
                total_tokens += retry_tokens
                print(f"[RETRY RESPONSE] Tokens: {retry_tokens} | Finish Reason: {retry_answer.get('finish_reason')}")
                print(f"[RETRY TEXT]\n{retry_answer['text']}")
            else:
                print("[VALIDATION PASSED] Output looks good.")
                
            success_count += 1
                
        except Exception as e:
            print(f"[ERROR] API Call failed: {e}")
            
        print("\n<EOT>\n" + "=" * 80)
        
    print("\n" + "#" * 80)
    print(f"EVALUATION COMPLETE")
    print(f"Total API Tokens Used: {total_tokens}")
    print(f"Tasks Successfully Processed: {success_count}/{len(tasks)}")
    print("#" * 80)


if __name__ == "__main__":
    main()
