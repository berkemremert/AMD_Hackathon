# Judging VM runs linux/amd64. Build and push that platform explicitly:
#   docker buildx build --platform linux/amd64 -t <registry>:latest --push .
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "openai>=1.30.0"

COPY main.py agent.py classifier.py llm.py solvers.py ./

# Optional hard override: pin every tier to one allowed model (guarded in
# llm.py — ignored if not in the harness's ALLOWED_MODELS). Empty by default;
# tier inference already routes correctly (gemma-4-31b-it strong,
# gemma-4-26b-a4b-it cheap, kimi-k2p7-code code, minimax-m3 avoided).
ARG PREFERRED_MODEL=
ENV PREFERRED_MODEL=${PREFERRED_MODEL}

# Harness mounts /input and /output and injects FIREWORKS_* at run time.
ENTRYPOINT ["python", "main.py"]
