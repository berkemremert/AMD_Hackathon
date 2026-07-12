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

from src.fireworks_client import chat

INPUT_PATH = Path(os.environ.get("TASK_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("TASK_OUTPUT_PATH", "/output/results.json"))

# Fallback to local dev env vars if ALLOWED_MODELS is not provided
if "ALLOWED_MODELS" in os.environ:
    models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    MODEL_CHEAP = next((m for m in models if "kimi" in m.lower()), models[-1] if models else "accounts/fireworks/models/kimi-k2p6")
    MODEL_EXPENSIVE = next((m for m in models if "minimax" in m.lower()), models[0] if models else "accounts/fireworks/models/kimi-k2p6")
else:
    MODEL_CHEAP = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/kimi-k2p6")
    MODEL_EXPENSIVE = os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p6")
ROUTER_MODE = os.environ.get("ROUTER_MODE", "finetuned")


def route(prompt: str) -> tuple[str, int]:
    """Returns (model_id, routing_tokens). routing_tokens is 0 for local routing."""
    if ROUTER_MODE == "always-cheap":
        return MODEL_CHEAP, 0
    if ROUTER_MODE == "always-expensive":
        return MODEL_EXPENSIVE, 0
    if ROUTER_MODE == "baseline":
        from src.baseline_router import classify
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
    from src.output_optimizer import detect_task_type, get_dynamic_limits
    from src import validator

    for task in tasks:
        task_type = detect_task_type(task["prompt"])
        

                
        try:
            if task_type == "entity_extraction" or task.get("category") == "entity_extraction":
                model, routing_tokens = MODEL_CHEAP, 0
            else:
                model, routing_tokens = route(task["prompt"])
                
            limits = get_dynamic_limits(task_type, task["prompt"])
            system_prompt = limits["system"]
            
            from src.local_compressor import optimize_prompt_for_api
            final_prompt = optimize_prompt_for_api(task["prompt"], task_type, limits.get("suffix", ""))
            
            answer = chat(
                model=model,
                prompt=final_prompt,
                max_tokens=limits["cap"],
                system_prompt=system_prompt,
                extra_params={"reasoning_effort": "none", "reasoning_history": "disabled"}
            )
            total_tokens += routing_tokens + answer.get("total_tokens", 0)
            
            ok, reason = validator.validate(task_type, task["prompt"], answer.get("text", ""), answer.get("finish_reason"))
            if not ok or not answer.get("text", "").strip():
                print(f"Validation failed for task {task['task_id']} ({reason}).", file=sys.stderr)
                
            results.append({"task_id": task["task_id"], "answer": str(answer.get("text", ""))})
        except Exception as e:
            print(f"[ERROR] API handling failed for task {task['task_id']}: {e}", file=sys.stderr)
            results.append({"task_id": task["task_id"], "answer": "Unable to process task."})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {OUTPUT_PATH}. Total tokens: {total_tokens}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"agent failed: {e}", file=sys.stderr)
        sys.exit(1)
