import os
import json
import requests
import time
import threading
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

KEYS_ENV = os.environ.get("FIREWORKS_API_KEYS")
if not KEYS_ENV:
    single_key = os.environ.get("FIREWORKS_API_KEY")
    if not single_key:
        print("Error: FIREWORKS_API_KEYS or FIREWORKS_API_KEY environment variable not set.")
        exit(1)
    API_KEYS = [single_key]
else:
    API_KEYS = [k.strip() for k in KEYS_ENV.split(",") if k.strip()]

if not API_KEYS:
    print("Error: No valid API keys found.")
    exit(1)

print(f"Using {len(API_KEYS)} API key(s) for concurrent generation.")

URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/deepseek-v4-pro"
OUTPUT_FILE = "data/category1_synthetic_v1.json"
TOTAL_PROMPTS = 100

TOPICS = [
    "quantum physics", "biology and genetics", "everyday mechanics", 
    "space and astronomy", "chemistry and elements", "geology and earth science", 
    "meteorology and weather", "human anatomy", "computer science algorithms",
    "economics and finance", "psychological phenomena", "botany", 
    "historical causes and effects", "linguistics", "engineering and architecture",
    "oceanography", "neuroscience", "materials science", "thermodynamics"
]

def get_words(text):
    return set(re.findall(r'\w+', text.lower()))

def get_prompt_template(topic, existing_prompts):
    existing_list = "\n".join([f"- {p}" for p in existing_prompts])
    
    return f"""You are an expert dataset creator for AI evaluation.
We are building an evaluation dataset for the "Factual Knowledge" category.
We need "How things work", "Why things happen", definitions, and explanation-style factual prompts.

Generate EXACTLY ONE new, distinct, and high-quality factual question about: {topic}.
It MUST require explanation, conceptual understanding, or causal reasoning.
Make sure it is highly specific and completely different from generic trivia.

CRITICAL: DO NOT generate any question that is similar to the following questions we already have:
{existing_list}

Output your response ONLY as a valid JSON object with the exact following schema:
{{
  "task_id": "synthetic_PLACEHOLDER",
  "category": "factual_knowledge",
  "prompt": "<your highly specific question here>",
  "reference_answer": "<core concept answer>",
  "answer_aliases": ["<alias 1>", "<alias 2>"],
  "difficulty": "<easy|medium|hard>",
  "source": "fireworks_synthetic",
  "source_id": "synthetic_PLACEHOLDER"
}}

Do not include markdown code blocks like ```json ... ```. Just return the raw JSON object. Make sure the JSON is valid and properly escaped.
"""

progress_lock = threading.Lock()
generated_items = []

def generate_single_prompt(api_key, index):
    topic = random.choice(TOPICS)
    
    with progress_lock:
        # Sample up to 50 existing prompts to save context, or all if less
        existing_prompts = [item['prompt'] for item in generated_items]
        if len(existing_prompts) > 50:
            existing_prompts = random.sample(existing_prompts, 50)
            
    payload = {
      "model": MODEL,
      "max_tokens": 2048,
      "top_p": 0.9,
      "temperature": 1.0, 
      "response_format": {"type": "json_object"},
      "messages": [
        {
          "role": "user",
          "content": get_prompt_template(topic, existing_prompts)
        }
      ]
    }

    headers = {
      "Accept": "application/json",
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.post(URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            print(f"[{index}/{TOTAL_PROMPTS}] API Error: {response.status_code}")
            return None

        data = response.json()
        content = data['choices'][0]['message']['content'].strip()

        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]

        content = content.strip()
        item = json.loads(content)
        
        item['category'] = "factual_knowledge"
        item['source'] = "fireworks_synthetic"
        
        return item
    except Exception as e:
        print(f"[{index}/{TOTAL_PROMPTS}] Exception occurred: {e}")
        return None

def worker_task(api_key):
    global generated_items
    
    while True:
        with progress_lock:
            if len(generated_items) >= TOTAL_PROMPTS:
                return
            index = len(generated_items) + 1
            
        print(f"Generating prompt using key ending in ...{api_key[-4:]}")
        
        success = False
        retries = 3
        while retries > 0 and not success:
            item = generate_single_prompt(api_key, index)
            if item:
                with progress_lock:
                    if len(generated_items) < TOTAL_PROMPTS:
                        prompt_text = item.get('prompt', '')
                        p_words = get_words(prompt_text)
                        
                        # Semantic Deduplication (Jaccard Similarity)
                        is_dup = False
                        for existing in generated_items:
                            e_words = get_words(existing['prompt'])
                            union = len(p_words.union(e_words))
                            if union == 0: continue
                            similarity = len(p_words.intersection(e_words)) / union
                            if similarity > 0.5: # More than 50% word overlap is a duplicate
                                is_dup = True
                                break
                                
                        if not is_dup:
                            seq_id = len(generated_items) + 1
                            item['task_id'] = f"cat1_synthetic_{seq_id:03d}"
                            item['source_id'] = f"synthetic_{seq_id:03d}"
                            
                            generated_items.append(item)
                            
                            with open(OUTPUT_FILE, "w") as f:
                                json.dump(generated_items, f, indent=2)
                            print(f"[{seq_id}/{TOTAL_PROMPTS}] Success: {item.get('prompt')}")
                            success = True
                        else:
                            print(f"Discarding semantic duplicate prompt: {prompt_text}")
                            retries -= 1
                    else:
                        return
            else:
                retries -= 1
                if retries > 0:
                    time.sleep(2)
        
        time.sleep(0.5)

def main():
    global generated_items
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r') as f:
                generated_items = json.load(f)
            print(f"Loaded {len(generated_items)} existing prompts from {OUTPUT_FILE}.")
        except:
            print(f"Could not parse {OUTPUT_FILE}, starting fresh.")

    if len(generated_items) >= TOTAL_PROMPTS:
        print("Already generated the required number of prompts.")
        return

    print(f"Starting generation of the remaining {TOTAL_PROMPTS - len(generated_items)} prompts...")
    
    with ThreadPoolExecutor(max_workers=len(API_KEYS)) as executor:
        futures = []
        for key in API_KEYS:
            futures.append(executor.submit(worker_task, key))
            
        for future in as_completed(futures):
            future.result()

    print(f"Finished. Total prompts in dataset: {len(generated_items)}")

if __name__ == "__main__":
    main()
