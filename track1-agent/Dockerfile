FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt download_models.py ./
RUN python3 -m venv /opt/venv && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 && \
    pip install -r requirements.txt

# This is the only model downloaded from Hugging Face. Save a standalone copy
# rather than relying on a mutable hub cache at judging time.
ENV LOCAL_EMBEDDING_MODEL=/models/all-MiniLM-L6-v2
RUN python3 download_models.py


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PATH=/opt/venv/bin:$PATH \
    LOCAL_EMBEDDING_MODEL=/models/all-MiniLM-L6-v2 \
    LOCAL_EMBEDDING_DEVICE=cpu \
    LOCAL_SUMMARY_MODE=compress_only \
    ROUTER_MODE=finetuned

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models

# Copy only files reachable from the judging entrypoint. Training, evaluation,
# demo code, datasets, and build-time downloader code stay out of the image.
COPY agent.py fireworks_client.py baseline_router.py local_solvers.py ./
COPY local_compressor.py output_optimizer.py validator.py ./
COPY router/infer_router.py ./router/infer_router.py
COPY router/checkpoints/router-distilbert/ ./router/checkpoints/router-distilbert/
COPY src/ ./src/

CMD ["python3", "agent.py"]
