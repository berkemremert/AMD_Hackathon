import os
import requests
import json
from dotenv import load_dotenv

load_dotenv("track1-agent/.env")
api_key = os.environ.get("FIREWORKS_API_KEY_1") or os.environ.get("FIREWORKS_API_KEY")

prompt = """You are an expert AI creating a challenging test dataset for evaluating other AI models.
Category: Condensing passages to a specific format or length constraint
Difficulty: hard

Generate 1 unique and diverse test query matching this category and difficulty.
For this query, provide the exact `prompt` the AI model will receive, and the `ground_truth` answer or grading rubric.

For "easy" difficulty, the tasks should be clear, direct, and straightforward.
For "hard" difficulty, the tasks should be complex, multi-layered, contain edge cases, or have stringent constraints.

Respond ONLY with a valid JSON object, starting with a "reasoning" key for your thoughts, followed by the "queries" array containing your single query. Format:
{
  "reasoning": "step-by-step thinking for generating this query...",
  "queries": [
    {"prompt": "...", "ground_truth": "..."}
  ]
}
"""

url = "https://api.fireworks.ai/inference/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
payload = {
    "model": "accounts/fireworks/models/kimi-k2p7-code",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 3000,
    "temperature": 0.8,
}

resp = requests.post(url, headers=headers, json=payload)
print(resp.json()["choices"][0]["message"]["content"])
