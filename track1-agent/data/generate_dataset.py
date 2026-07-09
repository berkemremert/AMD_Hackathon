"""Generates a synthetic dataset for the 8 hackathon capability categories.
Uses Fireworks API to generate diverse, novel queries dynamically.
Supports multiple API keys for concurrent generation.
"""
import os
import re
import json
import time
import requests
import concurrent.futures
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KEYS = []
if "FIREWORKS_API_KEY" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY"])
elif "FIREWORKS_API_KEY_1" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY_1"])

if not KEYS:
    raise ValueError("No Fireworks API keys found. Please set FIREWORKS_API_KEY in .env")

BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL_POOL = os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p7-code").split(",")

OUT_PATH = Path(__file__).parent / "queries_raw.json"

CATEGORIES = {
    "factual_knowledge": "Explaining concepts, definitions, and how things work.",
    "math_reasoning": "Multi-step arithmetic, percentages, word problems, projections.",
    "sentiment_classification": "Labelling sentiment and justifying the classification.",
    "text_summarization": "Condensing passages to a specific format or length constraint.",
    "named_entity_recognition": "Extracting and labelling entities (person, org, location, date).",
    "code_debugging": "Identifying bugs in code snippets and providing corrected implementations.",
    "logical_reasoning": "Constraint-based puzzles where all conditions must be satisfied."
}

PROMPT_GENERAL = """You are an expert AI creating a challenging test dataset for evaluating other AI models.
Category: {category_desc}
Difficulty: {difficulty}

Generate 1 unique and diverse test query matching this category and difficulty.
For this query, provide the exact `prompt` the AI model will receive, and the `ground_truth` answer or grading rubric.

For "easy" difficulty, the tasks should be clear, direct, and straightforward.
For "hard" difficulty, the tasks should be complex, multi-layered, contain edge cases, or have stringent constraints.

Respond ONLY with a valid JSON object containing the "queries" array. Do NOT include any reasoning, conversational text, or markdown formatting outside the JSON object.
IMPORTANT: Your entire response must be valid parseable JSON. Escape all newlines as `\n` and all backslashes as `\\\\`. Format:
{{
  "queries": [
    {{"prompt": "...", "ground_truth": "..."}}
  ]
}}
"""

PROMPT_CODE_GEN = """You are an expert AI creating a Python coding test dataset.
Category: Writing correct, well-structured Python functions from a spec.
Difficulty: {difficulty}

Generate 1 unique and diverse Python coding challenge matching this difficulty.
The `prompt` must ask the user to write a specific Python function.
The `ground_truth` must be an object containing `function_name` and a `tests` array.
Each test in `tests` must have `args` (an array of positional arguments) and `expected` (the expected return value, using valid JSON types).

For "easy" difficulty, tasks should be simple (e.g. palindromes, basic math, list filtering).
For "hard" difficulty, tasks should be algorithmic or complex (e.g. DP, trees, graphs, complex interval merges).

Respond ONLY with a valid JSON object containing the "queries" array. Do NOT include any reasoning, conversational text, or markdown formatting outside the JSON object.
IMPORTANT: Your entire response must be valid parseable JSON. Escape all newlines as `\n` and all backslashes as `\\\\`. Format:
{{
  "queries": [
    {{
      "prompt": "Write a Python function `is_even(n)` that returns True if n is even, else False.",
      "ground_truth": {{
        "function_name": "is_even",
        "tests": [
          {{"args": [2], "expected": true}},
          {{"args": [3], "expected": false}}
        ]
      }}
    }}
  ]
}}
"""

def clean_json_string(s: str) -> str:
    # Fix invalid JSON escapes (e.g., \times, \$, \frac) by double-escaping them
    return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', s)

def extract_json_array(text: str) -> list:
    # 1. Try extracting from markdown block
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = clean_json_string(match.group(1))
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "queries" in data:
                return data["queries"]
            elif isinstance(data, list):
                return data
        except Exception:
            pass
            
    # 2. Try parsing the whole thing
    text_clean = clean_json_string(text.strip())
    try:
        data = json.loads(text_clean)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        elif isinstance(data, list):
            return data
    except Exception:
        pass
        
    # 3. Regex fallback
    match = re.search(r"\{.*\}", text_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if "queries" in data:
                return data["queries"]
        except Exception:
            pass
            
    match = re.search(r"\[.*\]", text_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass
            
    raise ValueError(f"No JSON array found in response: {text[:200]}...")

def generate_batch(category: str, difficulty: str, api_key: str) -> list[dict]:
    if category == "code_generation":
        prompt = PROMPT_CODE_GEN.format(difficulty=difficulty)
    else:
        prompt = PROMPT_GENERAL.format(category_desc=CATEGORIES[category], difficulty=difficulty)
        
    model_to_use = random.choice(MODEL_POOL)
    
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20000,
        "temperature": 0.8,
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            parsed = extract_json_array(content)
            
            out = []
            for item in parsed:
                # Filter out dummy template responses
                if item.get("prompt") == "..." or item.get("ground_truth") == "...":
                    continue
                    
                # Ensure code_generation ground_truth is stringified
                if category == "code_generation":
                    gt = json.dumps(item["ground_truth"])
                else:
                    gt = str(item["ground_truth"])
                    
                out.append({
                    "category": category,
                    "difficulty_pool": difficulty,
                    "prompt": item["prompt"],
                    "ground_truth": gt
                })
            return out
        except Exception as e:
            print(f"Attempt {attempt+1} failed for {category} ({difficulty}): {e}")
            time.sleep(3 * (attempt + 1))
            
    return []

def worker(args):
    idx, category, difficulty = args
    api_key = KEYS[idx % len(KEYS)]
    print(f"Generating {category} ({difficulty}) using Key {idx % len(KEYS) + 1}...")
    return generate_batch(category, difficulty, api_key)

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for Track 1 Agent")
    parser.add_argument("--queries_per_pool", type=int, default=10, help="Target number of queries per category/difficulty pool")
    args = parser.parse_args()

    QUERIES_PER_BATCH = 1
    TARGET_PER_POOL = args.queries_per_pool
    
    existing = []
    if OUT_PATH.exists():
        try:
            existing = json.loads(OUT_PATH.read_text())
        except json.JSONDecodeError:
            pass

    counts = {}
    for q in existing:
        key = (q["category"], q["difficulty_pool"])
        counts[key] = counts.get(key, 0) + 1

    tasks = []
    task_idx = 0
    all_categories = list(CATEGORIES.keys()) + ["code_generation"]
    
    for cat in all_categories:
        for diff in ["easy", "hard"]:
            current_count = counts.get((cat, diff), 0)
            needed = max(0, TARGET_PER_POOL - current_count)
            batches_needed = (needed + QUERIES_PER_BATCH - 1) // QUERIES_PER_BATCH
            
            for _ in range(batches_needed):
                tasks.append((task_idx, cat, diff))
                task_idx += 1
                
    if not tasks:
        print(f"Dataset already complete with {len(existing)} queries. Nothing to do!")
        return
        
    max_workers = len(KEYS) * 4
    print(f"Resuming generation. Need {len(tasks)} more batches. Using {len(KEYS)} API key(s) and {max_workers} workers...")
    
    try:
        start_time = time.time()
        completed_tasks = 0
        total_tasks = len(tasks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(worker, t): t for t in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                t = future_to_task[future]
                completed_tasks += 1
                
                # Calculate ETA
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_tasks
                remaining = total_tasks - completed_tasks
                eta_str = time.strftime('%M:%S', time.gmtime(remaining * avg_time))
                
                try:
                    batch_results = future.result()
                    if batch_results:
                        existing.extend(batch_results)
                        # Reassign IDs cleanly
                        for i, q in enumerate(existing):
                            q["id"] = f"q{i:03d}"
                        # Save incrementally
                        OUT_PATH.write_text(json.dumps(existing, indent=2))
                        print(f"[{completed_tasks}/{total_tasks} | ETA: {eta_str}] Saved query for {t[1]} ({t[2]}). Dataset size: {len(existing)}")
                    else:
                        print(f"[{completed_tasks}/{total_tasks} | ETA: {eta_str}] Failed to generate for {t[1]} ({t[2]}).")
                except Exception as exc:
                    print(f"[{completed_tasks}/{total_tasks} | ETA: {eta_str}] Task {t[1]} ({t[2]}) generated an exception: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been saved and can be resumed later.")

    print(f"\nDone. Dataset currently has {len(existing)} queries. Saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
