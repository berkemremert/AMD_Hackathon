# TokenRouter

A token-efficient agent for the AMD Developer Hackathon (ACT II) — Track 1.
The contest ranks passing submissions by **fewest total Fireworks tokens**,
subject to an LLM-judge accuracy gate.

## How it works

Each task is answered in two steps, the first of which spends nothing:

1. **Zero-token classification** (`classifier.py`) — a single regex pass over
   the prompt assigns one of eight categories (factual, math, sentiment,
   summarization, NER, code-debug, code-gen, logic). No model call, so routing
   is free against the scored token budget.

2. **Deterministic solvers** (`solvers.py`) — for logic assignment puzzles and
   pure arithmetic, plain code answers with certainty at zero tokens. Every
   solver self-gates and defers to the model on the slightest ambiguity, so it
   can never turn a gettable task into a wrong answer.

3. **One tuned Fireworks call** (`agent.py` + `llm.py`) — each category carries
   a terse system prompt, a token cap, and a model tier. Tiers (`cheap`,
   `strong`, `code`) are inferred at runtime from whatever IDs arrive in
   `ALLOWED_MODELS`, never hardcoded. `reasoning_effort=none` suppresses hidden
   thinking tokens (which are scored); a blank or failed answer retries once on
   the opposite tier so a zero-scoring empty reply never ships.

`main.py` reads `/input/tasks.json`, answers tasks in a thread pool with a
deadline safely inside the 10-minute budget, and writes `/output/results.json`
with every `task_id` echoed exactly as given.

## Run it like the harness does

```bash
docker build -t tokenrouter .
docker run --rm \
  -v "$PWD/sample_input:/input:ro" -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  tokenrouter
```

Local dev without Docker: copy `.env.example` to `.env`, fill in your key, then
`INPUT_PATH=sample_input/tasks.json OUTPUT_PATH=out/results.json python main.py`.

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Entrypoint: I/O contract, thread pool, deadline |
| `classifier.py` | Zero-token category detection |
| `solvers.py` | Zero-token deterministic solvers (logic puzzles, arithmetic) |
| `agent.py` | Per-category prompt / token-cap / tier strategy |
| `llm.py` | Fireworks client, tier inference, token accounting |
| `Dockerfile` | `python:3.12-slim` + `openai` |
| `arsive/` | Earlier Go / local-model implementation, kept for reference |

## Submission

CI builds and pushes the linux/amd64 image to GHCR on every push to `main`.
Submit the image reference on the lablab form.
