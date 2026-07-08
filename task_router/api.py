import requests

def call_fireworks_api(prompt: str, api_key: str, base_url: str, model: str) -> str:
    """Makes the API request to Fireworks."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0, # Zero temp for deterministic, concise answers
        "max_tokens": 512
    }
    
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    
    return resp.json()['choices'][0]['message']['content']
