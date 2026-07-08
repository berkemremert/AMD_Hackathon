import os
import sys
import json
import argparse
from router import classify_task, maybe_solve_locally, build_minimal_prompt, post_process
from api import call_fireworks_api

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/input/tasks.json", help="Path to tasks.json")
    parser.add_argument("--output", default="/output/results.json", help="Path to results.json")
    args = parser.parse_args()

    # Read environment variables injected by harness
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL")
    allowed_models_str = os.environ.get("ALLOWED_MODELS", "")
    
    if not all([api_key, base_url, allowed_models_str]):
        print("Missing required environment variables. Aborting.")
        sys.exit(1)
        
    models = [m.strip() for m in allowed_models_str.split(",") if m.strip()]
    if not models:
        print("No ALLOWED_MODELS found. Aborting.")
        sys.exit(1)
        
    model = models[0] # Use the first allowed model by default
    
    # Read Tasks
    try:
        with open(args.input, "r") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Failed to read {args.input}: {e}")
        sys.exit(1)
        
    results = []
    
    for task in tasks:
        tid = task.get("task_id")
        prompt = task.get("prompt", "")
        
        # 1. Classify
        category = classify_task(prompt)
        
        # 2. Local Fallback (0 tokens)
        local_ans = maybe_solve_locally(category, prompt)
        if local_ans is not None:
            ans = local_ans
        else:
            # 3. Build optimized prompt
            opt_prompt = build_minimal_prompt(category, prompt)
            
            # 4. API Call
            try:
                raw_ans = call_fireworks_api(opt_prompt, api_key, base_url, model)
                # 5. Post Process
                ans = post_process(category, raw_ans)
            except Exception as e:
                ans = f"Error: {e}"
                
        results.append({
            "task_id": tid,
            "answer": ans
        })
        
    # Ensure output directory exists (mostly for local testing)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Write Results
    try:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Failed to write {args.output}: {e}")
        sys.exit(1)
        
    print(f"Successfully processed {len(tasks)} tasks.")
    sys.exit(0)

if __name__ == "__main__":
    main()
