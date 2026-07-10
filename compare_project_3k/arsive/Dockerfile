# Build for the judging VM: linux/amd64, compressed image must stay under 10GB.
#   docker buildx build --platform linux/amd64 -t <registry>/token-router:latest --push .
# The official llama.cpp images ship the full toolchain (~6GB); we build a
# lean static llama-server instead — final image ≈ model + ~200MB.

# INCLUDE_MODEL=false builds a weightless diagnostic image (~150MB vs ~3.2GB):
# a zero-byte placeholder ships at the model path and the agent degrades to
# Fireworks-only. Isolates registry-pull-size and llama-server variables.
ARG INCLUDE_MODEL=true

# --- Stage 1: Go binary ---
FROM --platform=$BUILDPLATFORM golang:1.26 AS gobuild
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
ARG TARGETARCH
RUN CGO_ENABLED=0 GOOS=linux GOARCH=$TARGETARCH go build -trimpath -ldflags="-s -w" -o /agent ./cmd/agent

# --- Stage 2: llama-server, static, portable CPU (no -march=native: the
# judge VM's CPU flags are unknown) ---
FROM debian:bookworm-slim AS llamabuild
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_TAG=b9890
RUN git clone --depth 1 --branch ${LLAMA_TAG} https://github.com/ggml-org/llama.cpp /llama
RUN cmake -S /llama -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF \
    && cmake --build /build --target llama-server -j"$(nproc)"

# --- Stage 3: model weights ---
# Grading box is 4 GB RAM / 2 vCPU: E2B Q4 (~3GB) fits alongside the KV cache
# and the agent; the E4B file alone (4.8GB) would not. See eval/PERF.md.
FROM alpine:3.20 AS model-true
ARG MODEL_URL=https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q4_K_XL.gguf
# --http1.1: HF's CDN intermittently resets HTTP/2 streams on multi-GB pulls
# (curl exit 92); retry-all-errors + resume make the pull deterministic.
RUN apk add --no-cache curl && \
    curl -fL --http1.1 --retry 5 --retry-all-errors -C - -o /model.gguf "$MODEL_URL"

# Weightless variant: zero-byte placeholder, startLocal treats it as absent.
FROM alpine:3.20 AS model-false
RUN touch /model.gguf

FROM model-${INCLUDE_MODEL} AS model

# --- Stage 4: runtime ---
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-minimal libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3
COPY --from=llamabuild /build/bin/llama-server /usr/local/bin/llama-server
COPY --from=gobuild /agent /usr/local/bin/agent
COPY --from=model /model.gguf /models/model.gguf

ENV LOCAL_MODEL_PATH=/models/model.gguf \
    LOCAL_SERVER_BIN=/usr/local/bin/llama-server \
    LOCAL_BASE_URL=http://127.0.0.1:8080

# Ladder knobs baked per-variant: the harness runs the image with ITS env
# only (FIREWORKS_*), so each submission rung is a build with different
# defaults. Defaults here match the code's own defaults.
ARG DEFAULT_LOCAL=true
ARG DEFAULT_LOCAL_CATEGORIES=""
ARG DEFAULT_WORKERS=4
ARG DEFAULT_REASONING_EFFORT=none
ARG DEFAULT_SDK_MIMIC=true
ARG DEFAULT_PREFIX_CACHE=true
ARG DEFAULT_REMOTE_CAPS=true
ARG DEFAULT_PROBE=false
ENV LOCAL=${DEFAULT_LOCAL} \
    LOCAL_CATEGORIES=${DEFAULT_LOCAL_CATEGORIES} \
    WORKERS=${DEFAULT_WORKERS} \
    REASONING_EFFORT=${DEFAULT_REASONING_EFFORT} \
    SDK_MIMIC=${DEFAULT_SDK_MIMIC} \
    PREFIX_CACHE=${DEFAULT_PREFIX_CACHE} \
    REMOTE_CAPS=${DEFAULT_REMOTE_CAPS} \
    PROBE=${DEFAULT_PROBE}

ENTRYPOINT ["/usr/local/bin/agent"]
