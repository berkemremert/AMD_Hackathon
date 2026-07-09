import json
import os
import time
from local_solvers import solve_ner
from fireworks_client import chat

import concurrent.futures

def evaluate_single_task(task, index, total, model_cheap, judge_model):
    prompt = task["prompt"]
    result = {"index": index, "pass": False, "reason": "", "fmt_tokens": 0, "judge_tokens": 0}
    
    try:
        # 1. Get our GLiNER 0-token answer
        raw_entities = solve_ner(prompt)
        
        # 2. Hybrid format using MODEL_CHEAP
        format_prompt = f"The user requested this extraction task:\n{prompt}\n\nOur local NER model extracted these preliminary entities: {raw_entities}\n\nYour job is to act as an intelligent refiner. Take these preliminary entities, format them exactly as requested in the task instructions, AND perfectly apply any specific inclusion/exclusion rules mentioned in the prompt (for example, stripping honorifics, ignoring relative dates, or merging nested entities). Output ONLY the final formatted result. Do not add any preamble."
        
        answer_resp = chat(model_cheap, format_prompt, max_tokens=800)
        answer = answer_resp["text"]
        result["fmt_tokens"] = answer_resp["total_tokens"]
        
        # 3. Ask the LLM Judge
        judge_prompt = f"""You are a strict automated grading system.
Task Instructions:
{prompt}

User's Output:
{answer}

Evaluate if the User's Output perfectly extracts the requested entities and matches the requested format.
You may think out loud, but you MUST end your response with exactly:
<VERDICT>YES</VERDICT> or <VERDICT>NO</VERDICT>"""

        judge_resp = chat(judge_model, judge_prompt, max_tokens=400, temperature=0.0)
        judge_text = judge_resp["text"].strip().lower()
        result["judge_tokens"] = judge_resp["total_tokens"]
        
        if "<verdict>yes</verdict>" in judge_text:
            result["pass"] = True
            print(f"Task {index}/{total}: ✅ Pass")
        else:
            reason = judge_text.split('<verdict>')[0].strip()
            result["reason"] = reason
            print(f"Task {index}/{total}: ❌ Fail\n      Judge Reason: {reason}")
            
    except Exception as e:
        print(f"Task {index}/{total}: ⚠️ API Error: {e}")
        
    return result

def evaluate_ner():
    data = json.load(open('data/labeled_dataset.json'))
    ner_tasks = [t for t in data if t['category'] == 'named_entity_recognition']
    
    # Test all 100 NER tasks
    sample = ner_tasks
    
    # Hardcode the official Track 1 judge model
    judge_model = "accounts/fireworks/models/glm-5p2"
    model_cheap = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/minimax-m3")
    
    print(f"Using Official Judge Model: {judge_model}")
    print(f"Evaluating all {len(sample)} NER tasks with 10 concurrent requests...")
    
    total = len(sample)
    correct = 0
    total_formatting_tokens = 0
    total_judge_tokens = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(evaluate_single_task, task, i+1, total, model_cheap, judge_model) for i, task in enumerate(sample)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res["pass"]:
                correct += 1
            total_formatting_tokens += res["fmt_tokens"]
            total_judge_tokens += res["judge_tokens"]
            
    print(f"\n=== Final Accuracy ===")
    print(f"{correct}/{total} Correct ({(correct/total)*100:.1f}%)")
    print(f"\n=== Token Usage ===")
    print(f"Hybrid Formatting Tokens Used (MODEL_CHEAP): {total_formatting_tokens:,}")
    print(f"Judge Evaluator Tokens Used (MODEL_JUDGE): {total_judge_tokens:,}")

if __name__ == "__main__":
    evaluate_ner()
