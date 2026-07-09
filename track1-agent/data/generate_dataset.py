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
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KEYS = []
if "FIREWORKS_API_KEY_1" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY_1"])
if "FIREWORKS_API_KEY_2" in os.environ:
    KEYS.append(os.environ["FIREWORKS_API_KEY_2"])

if not KEYS:
    if "FIREWORKS_API_KEY" in os.environ:
        KEYS.append(os.environ["FIREWORKS_API_KEY"])
    else:
        raise ValueError("No Fireworks API keys found. Please set FIREWORKS_API_KEY_1 and FIREWORKS_API_KEY_2 in .env")

BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
MODEL = os.environ.get("MODEL_EXPENSIVE", "accounts/fireworks/models/kimi-k2p7-code")

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

Generate 5 unique and diverse test queries matching this category and difficulty.
For each query, provide the exact `prompt` the AI model will receive, and the `ground_truth` answer or grading rubric.

For "easy" difficulty, the tasks should be clear, direct, and straightforward.
For "hard" difficulty, the tasks should be complex, multi-layered, contain edge cases, or have stringent constraints.

Respond ONLY with a valid JSON array of objects in this exact format, with no backticks, markdown, or extra text:
[
  {{"prompt": "...", "ground_truth": "..."}}
]
"""

PROMPT_CODE_GEN = """You are an expert AI creating a Python coding test dataset.
Category: Writing correct, well-structured Python functions from a spec.
Difficulty: {difficulty}

Generate 5 unique and diverse Python coding challenges matching this difficulty.
The `prompt` must ask the user to write a specific Python function.
The `ground_truth` must be an object containing `function_name` and a `tests` array.
Each test in `tests` must have `args` (an array of positional arguments) and `expected` (the expected return value, using valid JSON types).

For "easy" difficulty, tasks should be simple (e.g. palindromes, basic math, list filtering).
For "hard" difficulty, tasks should be algorithmic or complex (e.g. DP, trees, graphs, complex interval merges).

Respond ONLY with a valid JSON array of objects in this exact format, with no backticks, markdown, or extra text:
[
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
"""

def extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text[:200]}...")
    return json.loads(match.group(0))

def generate_batch(category: str, difficulty: str, api_key: str) -> list[dict]:
    if category == "code_generation":
        prompt = PROMPT_CODE_GEN.format(difficulty=difficulty)
    else:
        prompt = PROMPT_GENERAL.format(category_desc=CATEGORIES[category], difficulty=difficulty)
        
    url = f"{BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3000,
        "temperature": 0.8,
    }
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            
            parsed = extract_json_array(content)
            
            out = []
            for item in parsed:
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
    # Number of batches per category/difficulty (5 queries per batch)
    BATCHES = 2 
    
    tasks = []
    task_idx = 0
    all_categories = list(CATEGORIES.keys()) + ["code_generation"]
    
    for cat in all_categories:
        for diff in ["easy", "hard"]:
            for _ in range(BATCHES):
                tasks.append((task_idx, cat, diff))
                task_idx += 1
                
    results = []
    
    # 4 concurrent workers per API key
    max_workers = len(KEYS) * 4
    print(f"Starting generation with {len(KEYS)} API key(s) and {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_results in executor.map(worker, tasks):
            results.extend(batch_results)
            
    # Assign sequential IDs
    for i, q in enumerate(results):
        q["id"] = f"q{i:03d}"
        
    if results:
        OUT_PATH.write_text(json.dumps(results, indent=2))
        print(f"\nSuccessfully generated {len(results)} queries. Saved to {OUT_PATH}")
    else:
        print("\nFailed to generate any queries.")

if __name__ == "__main__":
    main()
