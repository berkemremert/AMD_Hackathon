import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

url = f"{BASE_URL}/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": "accounts/fireworks/models/kimi-k2p6",
    "messages": [{"role": "user", "content": "Explain what 2+2 is in one sentence."}],
    "max_tokens": 100,
    "temperature": 0,
    "reasoning_effort": "low",
    "reasoning_history": "disabled"
}

resp = requests.post(url, headers=headers, json=payload)
print(json.dumps(resp.json(), indent=2))
