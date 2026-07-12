import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
)

response = client.chat.completions.create(
    model="accounts/fireworks/models/minimax-m2p7",
    messages=[{"role": "user", "content": "Explain what 2+2 is in one sentence."}],
    max_tokens=100,
    temperature=0,
    extra_body={"reasoning_effort": "low"}
)

print("Raw response:")
print(response.model_dump_json(indent=2))
