FROM python:3.11-slim

WORKDIR /app

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt hf-transfer

# Enable blazing fast downloads (silences the HF_HUB_ENABLE_HF_TRANSFER warning)
ENV HF_XET_HIGH_PERFORMANCE=1

# Create HF cache directory
ENV HF_HOME=/models/huggingface
RUN mkdir -p $HF_HOME

# Copy Python files required for model downloading
COPY download_models.py local_solvers.py output_optimizer.py ./

# Bake Hugging Face models into the Docker image so they don't download at runtime
RUN python3 download_models.py

# Prevent Hugging Face from checking the internet for model updates at runtime
ENV HF_HUB_OFFLINE=1

# Local Summarization Model is disabled to save RAM and build time.
# ARG LOCAL_SUMMARY_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
# ENV LOCAL_SUMMARY_MODEL=${LOCAL_SUMMARY_MODEL}

# Copy all python files, the router model directory, and test data
COPY *.py ./
COPY router/ ./router/
COPY data/ ./data/
COPY src/ ./src/

ENV ROUTER_MODE=finetuned

# Run the evaluation script by default to demonstrate the agent
CMD ["python3", "agent.py"]
