"""Quick smoke-test for the Fireworks AI chat completions API.

Sends a trivial JSON-mode request to verify that the API key and
endpoint are working. Requires the ``FIREWORKS_API_KEYS`` or
``FIREWORKS_API_KEY`` environment variable to be set.
"""

import os
import sys
import requests
import json

keys_env = os.environ.get("FIREWORKS_API_KEYS")
if not keys_env:
    keys_env = os.environ.get("FIREWORKS_API_KEY")
if not keys_env:
    print("Error: Set FIREWORKS_API_KEYS or FIREWORKS_API_KEY environment variable.")
    sys.exit(1)

api_key = keys_env.split(",")[0].strip()
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"

payload = {
    "model": MODEL,
    "max_tokens": 100,
    "response_format": {"type": "json_object"},
    "messages": [
        {"role": "user", "content": "Return a JSON object with 'hello': 'world'."}
    ]
}

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

print("Testing API...")
response = requests.post(URL, headers=headers, json=payload, timeout=30)
print("Status Code:", response.status_code)
if response.status_code != 200:
    print("Error:", response.text)
else:
    print("Success:", response.json()['choices'][0]['message']['content'])
