import json
import os
from fireworks_client import chat
from local_solvers import solve_ner

data = json.load(open('data/labeled_dataset.json'))
task = [t for t in data if t['category'] == 'named_entity_recognition'][1] # Task 2
prompt = task["prompt"]

raw_entities = solve_ner(prompt)
print(f"RAW ENTITIES:\n{raw_entities}\n")

model_cheap = os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/minimax-m3")
format_prompt = f"The user requested this task:\n{prompt}\n\nI have already extracted the entities for you: {raw_entities}\n\nYour ONLY job is to take these exact entities and format them exactly as requested in the task instructions (e.g., as tuples, lists, or custom JSON keys). Do not add any preamble. Output ONLY the final formatted result."

answer_resp = chat(model_cheap, format_prompt, max_tokens=800)
answer = answer_resp["text"]
print(f"HYBRID ANSWER:\n{answer}\n")

judge_model = "accounts/fireworks/models/glm-5p2"
judge_prompt = f"""You are a strict automated grading system.
Task Instructions:
{prompt}

User's Output:
{answer}

Evaluate if the User's Output perfectly extracts the requested entities and matches the requested format.
You may think out loud, but you MUST end your response with exactly:
<VERDICT>YES</VERDICT> or <VERDICT>NO</VERDICT>"""

judge_resp = chat(judge_model, judge_prompt, max_tokens=200, temperature=0.0)
print(f"JUDGE VERDICT:\n{judge_resp['text']}")
