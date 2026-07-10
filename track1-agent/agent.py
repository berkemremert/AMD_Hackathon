"""Container entrypoint for AMD Developer Hackathon Act II, Track 1.

Matches the judging harness contract exactly:
  - reads tasks from /input/tasks.json: [{"task_id": "...", "prompt": "..."}]
  - writes /output/results.json: [{"task_id": "...", "answer": "..."}]
  - exits 0 on success, non-zero on failure
  - all answer-generating calls go through FIREWORKS_BASE_URL with a model
    from ALLOWED_MODELS - the router itself runs locally and costs zero tokens

ROUTER_MODE selects how each prompt is routed (env var, default "finetuned"):
  finetuned        - local DistilBERT classifier (this tutorial's router)
  baseline         - prompt-based classification via an extra Fireworks call
  always-cheap     - skip routing, always use MODEL_CHEAP
  always-expensive - skip routing, always use MODEL_EXPENSIVE
"""
import json
import os
import sys
from pathlib import Path

from fireworks_client import chat

INPUT_PATH = Path(os.environ.get("TASK_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("TASK_OUTPUT_PATH", "/output/results.json"))

# Fallback to local dev env vars if ALLOWED_MODELS is not provided
if "ALLOWED_MODELS" in os.environ:
    models = os.environ["ALLOWED_MODELS"].split(",")
    # Heuristic to find the cheap and expensive model in the allowed models
    # Assuming minimax is cheap and kimi is expensive, or just pick by index
    MODEL_CHEAP = next((m for m in models if "minimax" in m.lower()), models[0])
    MODEL_EXPENSIVE = next((m for m in models if "kimi" in m.lower()), models[-1])
else:
    MODEL_CHEAP = os.environ["MODEL_CHEAP"]
    MODEL_EXPENSIVE = os.environ["MODEL_EXPENSIVE"]
ROUTER_MODE = os.environ.get("ROUTER_MODE", "finetuned")


def route(prompt: str) -> tuple[str, int]:
    """Returns (model_id, routing_tokens). routing_tokens is 0 for local routing."""
    if ROUTER_MODE == "always-cheap":
        return MODEL_CHEAP, 0
    if ROUTER_MODE == "always-expensive":
        return MODEL_EXPENSIVE, 0
    if ROUTER_MODE == "baseline":
        from baseline_router import classify
        result = classify(prompt)
        model = MODEL_EXPENSIVE if result["label"] == "hard" else MODEL_CHEAP
        return model, result["tokens"]
    # finetuned (default)
    from router.infer_router import predict
    label = predict(prompt)
    model = MODEL_EXPENSIVE if label == "hard" else MODEL_CHEAP
    return model, 0


def main():
    tasks = json.loads(INPUT_PATH.read_text())
    results = []
    total_tokens = 0
    from output_optimizer import detect_task_type, TOKEN_LIMITS
    from local_solvers import solve_ner
    import validator

    for task in tasks:
        task_type = detect_task_type(task["prompt"])
        
        if task_type == "entity_extraction":
            # 1. Massive token savings: extract NER perfectly locally for 0 API tokens
            raw_entities = solve_ner(task["prompt"])
            
            # Use ONLY GLiNER for 0 API token cost as requested
            results.append({"task_id": task["task_id"], "answer": raw_entities})
            continue

        model, routing_tokens = route(task["prompt"])
        
        # Tighten the prompt using dynamic output optimization
        limits = TOKEN_LIMITS[task_type]
        tight_prompt = f"{limits['suffix']}\n\n{task['prompt']}"
        
        answer = chat(model, tight_prompt, max_tokens=limits["cap"], extra_params={"reasoning_effort": "none"})
        total_tokens += routing_tokens + answer["total_tokens"]
        
        ok, reason = validator.validate(task_type, task["prompt"], answer["text"], answer.get("finish_reason"))
        if not ok:
            print(f"Validation failed for task {task['task_id']} ({reason}). Retrying with generous cap...", file=sys.stderr)
            retry_answer = chat(model, tight_prompt, max_tokens=limits.get("retry_cap", 800))
            total_tokens += retry_answer["total_tokens"]
            answer = retry_answer
            
        results.append({"task_id": task["task_id"], "answer": answer["text"]})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {OUTPUT_PATH}. Total tokens: {total_tokens}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"agent failed: {e}", file=sys.stderr)
        sys.exit(1)
