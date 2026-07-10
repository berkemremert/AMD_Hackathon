FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the runtime pipeline — no tests, no eval/bakeoff tooling, no .env.
COPY main.py classifier.py router.py fireworks_client.py validator.py ./

# All config (FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS) comes
# from runtime env vars injected by the harness. Nothing is baked in.
ENTRYPOINT ["python", "main.py"]
