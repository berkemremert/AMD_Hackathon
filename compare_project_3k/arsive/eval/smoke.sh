#!/usr/bin/env bash
# Live Fireworks smoke test — run BEFORE a real submission so a config problem
# never burns a leaderboard slot. Validates the exact client code the agent
# uses (model selection, reasoning_effort, prefix-cache header) against the
# real endpoint, then runs the official practice tasks end-to-end.
#
#   1. Put your Fireworks creds in a .env file (gitignored) at the repo root:
#        FIREWORKS_API_KEY=fw_...
#        FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
#        ALLOWED_MODELS=accounts/fireworks/models/gemma-2-9b-it,...
#      (Use whatever model IDs your Fireworks account/base_url actually serves.
#       If the harness gives short IDs like "gemma-4-31b-it", those may only
#       exist behind the harness proxy — for dev use your account's real paths.)
#   2. eval/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a; . ./.env; set +a
fi
: "${FIREWORKS_API_KEY:?set it in .env}"
: "${FIREWORKS_BASE_URL:?set it in .env}"
: "${ALLOWED_MODELS:?set it in .env}"

echo "### 1/2 — client path (Pick + reasoning_effort + prefix-cache)"
go run ./cmd/smoke

echo
echo "### 2/2 — official practice.json end-to-end (remote-only, LOCAL=0)"
OUT_DIR="${OUT_DIR:-eval/out-smoke}"; mkdir -p "$OUT_DIR"
LOCAL=0 INPUT_PATH=eval/practice.json OUTPUT_PATH="$OUT_DIR/results.json" \
  go run ./cmd/agent 2> >(tee "$OUT_DIR/trace.log" >&2)

echo
echo "### answers:"
python3 - "$OUT_DIR/results.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))
assert all("task_id" in r and "answer" in r for r in res), "INVALID SCHEMA"
for r in res:
    print(f"  {r['task_id']:12} {r['answer'][:90]!r}")
print(f"\n{len(res)} answers, valid schema, none empty:",
      all(r["answer"].strip() for r in res))
PY
echo
echo "Compare answers to eval/expected-practice.json by eye; check the token"
echo "summary in the trace above. If all green, this image is submission-ready."
