# Bastion

![Bastion cover](assets/bastion-cover.png)

Track 1 agent for the AMD Developer Hackathon: ACT II.

Bastion is an accuracy-first, token-efficient general-purpose AI agent. It handles the official Track 1 task format, uses only models from `ALLOWED_MODELS`, sends all Fireworks calls through `FIREWORKS_BASE_URL`, and writes the required `/output/results.json`.

The implementation package is named `frugalrouter`; the submitted project name is Bastion.

## Official Track 1 Contract

The submitted container must:

1. Read `/input/tasks.json` on startup.
2. Write `/output/results.json` before exiting.
3. Read `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` from the environment.
4. Use only allowed Fireworks models.
5. Exit with code 0 on success.

Expected input:

```json
[
  { "task_id": "t1", "prompt": "Summarise the following text in one sentence: ..." }
]
```

Expected output:

```json
[
  { "task_id": "t1", "answer": "..." }
]
```

## Strategy

```text
task -> classify category -> deterministic shortcut if very safe
     -> choose allowed Fireworks model
     -> concise prompt through FIREWORKS_BASE_URL
     -> write official answer JSON
     -> log route/model/token metadata separately
```

The agent covers the eight Track 1 categories: factual Q&A, math reasoning, sentiment, summarization, NER, code debugging, logic puzzles, and code generation.

Deterministic shortcuts only fire when they are provably safe: sentiment needs a clear multi-keyword majority with no negation, math needs a purely computational prompt, and exact-response instructions must not offer alternatives. Everything else escalates to Fireworks with per-category model preference (Gemma models where sensible, `kimi-k2p7-code` for code) and per-category token caps.

Remote calls are resilient by design: transient errors (429/5xx) retry with exponential backoff, a persistently failing model fails over to the next allowed model, 404 model names advance immediately, and an empty answer at the token cap is retried once with a larger cap. A task never crashes the batch.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Local dry run without Fireworks:

```powershell
python -m frugalrouter.cli --tasks data/sample_tasks.json --output outputs/sample_debug.jsonl --no-remote --debug-output
```

Local run with Fireworks:

```powershell
$env:ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"
$env:FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
python -m frugalrouter.cli --tasks data/sample_tasks.json --output outputs/sample_results.json
python -m frugalrouter.report --log logs/decisions.jsonl
```

Use `FRUGAL_WORKERS` or `--workers` to control parallel task execution. The default is `4`.

## Local Evaluation

Run the curated local eval without spending Fireworks tokens:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1
```

Run with Fireworks enabled for development:

```powershell
$env:FIREWORKS_API_KEY=[Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY","User")
$env:FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
$env:FIREWORKS_MODEL_ID="accounts/fireworks/models/gpt-oss-120b"
powershell -ExecutionPolicy Bypass -File scripts/run_eval.ps1 -Remote -Workers 1 -Delay 2
```

Personal Fireworks keys have tight rate limits; `-Workers 1 -Delay 2` paces requests so 429s do not eat the run.

Reports are written to `reports/eval_report.json` and `reports/eval_results.json`. The report tracks graded accuracy, per-category accuracy, token spend, classifier accuracy, truncated answers (`finish_reason=length`), and, most importantly, `local_wrong_task_ids`: local-shortcut answers that failed grading. That list must stay empty; a wrong local answer is an accuracy-gate risk by definition.

## Docker

Build a linux/amd64 image:

```powershell
docker buildx build --platform linux/amd64 -t bastion:track1 .
```

Local container smoke test:

```powershell
New-Item -ItemType Directory -Force -Path input,output | Out-Null
Copy-Item data/sample_tasks.json input/tasks.json
docker run --rm `
  -e FIREWORKS_API_KEY=$env:FIREWORKS_API_KEY `
  -e FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" `
  -e ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4" `
  -v ${PWD}/input:/input `
  -v ${PWD}/output:/output `
  bastion:track1
```

For submission, build, verify, and publish in one step (the push only happens with `-Push`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza          # build + verify only
powershell -ExecutionPolicy Bypass -File scripts/build_submission.ps1 -Registry docker.io/kgotsomsiza -Push    # publish for submission
```

The script runs the test suite, builds the linux/amd64 image, and verifies the container contract (reads `/input/tasks.json`, writes `/output/results.json`, exits 0) before any push.

Published submission image:

```text
docker.io/kgotsomsiza/bastion:track1
```

## Project Layout

- `frugalrouter/cli.py` - official Track 1 runner.
- `frugalrouter/router.py` - route decision logic.
- `frugalrouter/task_classifier.py` - category classifier.
- `frugalrouter/model_policy.py` - allowed-model selection.
- `frugalrouter/providers/local.py` - deterministic zero-token shortcuts.
- `frugalrouter/providers/fireworks.py` - Fireworks judging-proxy client.
- `frugalrouter/decision_log.py` - JSONL decision logging.
- `frugalrouter/report.py` - summary report for route decisions.
- `config/models.json` - model routing and token caps.
- `docs/track1_battle_plan.md` - solo-builder plan for Track 1.

## Troubleshooting

If Docker Desktop dies at startup with `remove ...: The file cannot be accessed by the system`, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/fix_docker.ps1
```

If the Twitch web player buffers or fails, launch the lablab.ai Twitch stream through VLC:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/watch_lablab_twitch.ps1
```
