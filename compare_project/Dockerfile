FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FRUGAL_ALLOW_REMOTE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY frugalrouter ./frugalrouter
COPY config ./config

RUN python -m pip install --no-cache-dir -e .

CMD ["python", "-m", "frugalrouter.cli"]
