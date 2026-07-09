import json
import os
import time
from local_solvers import solve_ner
from fireworks_client import chat

def evaluate_ner():
    data = json.load(open('data/labeled_dataset.json'))
    ner_tasks = [t for t in data if t['category'] == 'named_entity_recognition']
    
    # Test all 100 NER tasks
    sample = ner_tasks
    
    # Hardcode the official Track 1 judge model
    judge_model = "accounts/fireworks/models/glm-5p2"
    print(f"Using Official Judge Model: {judge_model}")
    print(f"Evaluating all {len(sample)} NER tasks...")
    
    correct = 0
    total = len(sample)
    total_formatting_tokens = 0
    total_judge_tokens = 0
    
    for i, task in enumerate(sample):
        prompt = task["prompt"]
        
        # 1. Get our GLiNER 0-token answer
        raw_entities = solve_ner(prompt)
        
        # 2. Hybrid format using MODEL_CHEAP
        model_cheap = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/minimax-m3")
        format_prompt = f"""The user requested this task:
{prompt}

I used a zero-shot model to extract the initial entities: {raw_entities}

Your job is to act as the final refiner and formatter:
1. Filter out any common nouns (e.g., "new campus", "headquarters", "office", "store") from the extracted entities. Only keep true Proper Named Entities.
2. Ensure the label strings EXACTLY match the labels requested in the prompt (e.g., if the prompt asks for "org", change "Organization" to "org").
3. Format the final entities EXACTLY as requested in the task instructions (e.g., as tuples, lists, or custom JSON keys).

Do not add any preamble. Output ONLY the final formatted result."""
        
        answer_resp = chat(model_cheap, format_prompt, max_tokens=800)
        answer = answer_resp["text"]
        total_formatting_tokens += answer_resp["total_tokens"]
        
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
            # We give it 400 tokens so it has plenty of room to write its essay before printing the verdict
            judge_resp = chat(judge_model, judge_prompt, max_tokens=400, temperature=0.0)
            judge_text = judge_resp["text"].strip().lower()
            total_judge_tokens += judge_resp["total_tokens"]
            
            if "<verdict>yes</verdict>" in judge_text:
                print(f"Task {i+1}/{total}: ✅ Pass")
                correct += 1
            else:
                print(f"Task {i+1}/{total}: ❌ Fail")
                print(f"      Judge Reason: {judge_text.split('<verdict>')[0].strip()}")
        except Exception as e:
            print(f"Task {i+1}/{total}: ⚠️ API Error: {e}")
            
    print(f"\n=== Final Accuracy ===")
    print(f"{correct}/{total} Correct ({(correct/total)*100:.1f}%)")
    print(f"\n=== Token Usage ===")
    print(f"Hybrid Formatting Tokens Used (MODEL_CHEAP): {total_formatting_tokens:,}")
    print(f"Judge Evaluator Tokens Used (MODEL_JUDGE): {total_judge_tokens:,}")

if __name__ == "__main__":
    evaluate_ner()
