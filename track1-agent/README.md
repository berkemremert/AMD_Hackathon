# Track 1: Token-Efficient General-Purpose Agent

This is the AMD Developer Hackathon Track 1 submission. The container reads
`/input/tasks.json`, handles safe task classes locally, sends every unresolved
task to an allowed Kimi model, and writes `/output/results.json`.

## Architecture

```text
agent.py
  └── src/track1_agent/pipeline.py
        ├── output_optimizer.py   task detection and output limits
        ├── local_solvers.py      math, NER, sentiment, small logic puzzles
        ├── local_compressor.py   API prompt compression
        └── fireworks_client.py   Fireworks HTTP client
```

The production pipeline has three rules:

1. A supported local solver returns the answer with zero API tokens.
2. Otherwise exactly one answer-generation request is sent to Kimi.
3. The response is returned as-is; there is no validation-triggered retry.

`eval_agent.py` reuses the same `TaskProcessor`, so evaluation and submission
cannot silently diverge.

## Required environment variables

- `FIREWORKS_API_KEY`
- `FIREWORKS_BASE_URL`
- `ALLOWED_MODELS` — must contain a Kimi model ID

Optional local settings:

- `TASK_INPUT_PATH` and `TASK_OUTPUT_PATH`
- `MODEL_API` to select one of multiple allowlisted Kimi models
- `MODEL_JUDGE` for the development evaluator only

## Build and run

```bash
docker build --platform linux/amd64 -t track1-agent .
docker run --rm \
  --env-file .env \
  -v "$PWD/test_files/fixtures:/input:ro" \
  -v "$PWD/test_files/output:/output" \
  track1-agent
```

## Development checks

```bash
python3 -m unittest discover -s test_files -v
python3 eval_agent.py
```

The evaluator makes additional judge calls for development metrics. Those
calls are not part of the submitted container path.
