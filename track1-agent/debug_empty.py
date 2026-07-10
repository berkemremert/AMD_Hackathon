import json, os
from local_summary_compressor import optimize_prompt_for_api
from fireworks_client import chat
from output_optimizer import TOKEN_LIMITS

data = json.load(open('data/labeled_dataset.json'))
t = [x for x in data if x['category'] == 'text_summarization'][0]

prompt = t['prompt']
suffix = TOKEN_LIMITS['summarization']['suffix']
tight_prompt = optimize_prompt_for_api(prompt, 'summarization', suffix)

print("--- TIGHT PROMPT ---")
print(tight_prompt)

resp = chat(os.environ.get("MODEL_CHEAP", "accounts/fireworks/models/minimax-m3"), tight_prompt, max_tokens=128)
print("--- RESPONSE ---")
print(resp)
