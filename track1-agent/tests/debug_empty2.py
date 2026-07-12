import json, os, requests
from dotenv import load_dotenv
load_dotenv()
from src.local_compressor import optimize_prompt_for_api
from src.output_optimizer import TOKEN_LIMITS

data = json.load(open('data/labeled_dataset.json'))
t = [x for x in data if x['category'] == 'text_summarization'][0]

prompt = t['prompt']
suffix = TOKEN_LIMITS['summarization']['suffix']
tight_prompt = optimize_prompt_for_api(prompt, 'summarization', suffix)

BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
headers = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}", "Content-Type": "application/json"}
payload = {
    "model": "accounts/fireworks/models/minimax-m3",
    "messages": [{"role": "user", "content": tight_prompt}],
    "max_tokens": 128,
}
resp = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
print(json.dumps(resp.json(), indent=2))
