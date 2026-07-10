FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt hf-transfer

# Enable blazing fast downloads (silences the HF_HUB_ENABLE_HF_TRANSFER warning)
ENV HF_XET_HIGH_PERFORMANCE=1

# Copy all python files, the router model directory, and test data
COPY *.py ./
COPY router/ ./router/
COPY data/ ./data/
# Accept HF_TOKEN as a build argument to speed up downloads
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Bake Hugging Face models into the Docker image so they don't download at runtime
RUN python3 download_models.py
ENV ROUTER_MODE=finetuned

# Run the evaluation script by default to demonstrate the agent
CMD ["python3", "eval_agent.py"]
