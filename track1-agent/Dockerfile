FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt download_models.py ./
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt hf-transfer \
    && mkdir -p /models/huggingface

ENV HF_HOME=/models/huggingface \
    HF_HUB_OFFLINE=0 \
    HF_XET_HIGH_PERFORMANCE=1

# Models are downloaded at build time; runtime inference is fully offline.
RUN python3 download_models.py
ENV HF_HUB_OFFLINE=1

COPY agent.py fireworks_client.py local_compressor.py output_optimizer.py ./
COPY src/__init__.py ./src/
COPY src/track1_agent/ ./src/track1_agent/

CMD ["python3", "agent.py"]
