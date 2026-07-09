FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fireworks_client.py baseline_router.py agent.py ./
COPY router/ ./router/

ENV ROUTER_MODE=finetuned

CMD ["python3", "agent.py"]
