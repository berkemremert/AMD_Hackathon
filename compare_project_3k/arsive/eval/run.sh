#!/usr/bin/env bash
# Eval harness (Fireworks-only architecture).
#
#   MOCK=1 eval/run.sh                 # offline contract test against the mock proxy
#   FIREWORKS_API_KEY=... FIREWORKS_BASE_URL=... ALLOWED_MODELS=... eval/run.sh   # live
#
# Grading happens afterwards: Claude judges results.json against expected*.json.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${OUT_DIR:-eval/out}"
mkdir -p "$OUT_DIR"

MOCK_PID=""
if [[ "${MOCK:-0}" == "1" ]]; then
  python3 eval/mock_fireworks.py &
  MOCK_PID=$!
  trap '[[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true' EXIT
  export FIREWORKS_BASE_URL="http://127.0.0.1:18080"
  export FIREWORKS_API_KEY="mock"
  export ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
  sleep 0.5
fi

START=$(date +%s)
INPUT_PATH="${INPUT_PATH:-eval/tasks.json}" OUTPUT_PATH="$OUT_DIR/results.json" \
  go run ./cmd/agent 2> >(tee "$OUT_DIR/trace.log" >&2)
echo "elapsed: $(( $(date +%s) - START ))s"
echo "results: $OUT_DIR/results.json  trace: $OUT_DIR/trace.log"
