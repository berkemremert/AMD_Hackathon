import json
import os
import time
from local_solvers import solve_ner
from fireworks_client import chat

def evaluate_ner():
    data = json.load(open('data/labeled_dataset.json'))
    ner_tasks = [t for t in data if t['category'] == 'named_entity_recognition']
    
    # Let's test the first 20 to get a solid sample without burning too much time
    sample = ner_tasks[:20]
    
    # Hardcode the official Track 1 judge model
    judge_model = "accounts/fireworks/models/glm-5p2"
    print(f"Using Official Judge Model: {judge_model}")
    print(f"Evaluating {len(sample)} NER tasks...")
    
    correct = 0
    total = len(sample)
    
    for i, task in enumerate(sample):
        prompt = task["prompt"]
        
        # 1. Get our GLiNER 0-token answer
        raw_entities = solve_ner(prompt)
        
        # 2. Hybrid format using MODEL_CHEAP
        model_cheap = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/minimax-m3")
        format_prompt = f"The user requested this task:\n{prompt}\n\nI have already extracted the entities for you: {raw_entities}\n\nYour ONLY job is to take these exact entities and format them exactly as requested in the task instructions (e.g., as tuples, lists, or custom JSON keys). Do not add any preamble. Output ONLY the final formatted result."
        
        answer_resp = chat(model_cheap, format_prompt, max_tokens=800)
        answer = answer_resp["text"]
        
        # 3. Ask the LLM Judge
        judge_prompt = f"""You are a strict automated grading system.
Task Instructions:
{prompt}

User's Output:
{answer}

Evaluate if the User's Output perfectly extracts the requested entities and matches the requested format.
You may think out loud, but you MUST end your response with exactly:
<VERDICT>YES</VERDICT> or <VERDICT>NO</VERDICT>"""

        try:
            judge_resp = chat(judge_model, judge_prompt, max_tokens=200, temperature=0.0)
            judge_text = judge_resp["text"].strip().lower()
            
            if "<verdict>yes</verdict>" in judge_text:
                print(f"Task {i+1}/{total}: ✅ Pass")
                correct += 1
            else:
                print(f"Task {i+1}/{total}: ❌ Fail")
                print(f"      Judge Reason: {judge_text.split('<verdict>')[0].strip()}")
        except Exception as e:
            print(f"Task {i+1}/{total}: ⚠️ API Error: {e}")
            
    print(f"\n=== Final Accuracy ===\n{correct}/{total} Correct ({(correct/total)*100:.1f}%)")

if __name__ == "__main__":
    evaluate_ner()
