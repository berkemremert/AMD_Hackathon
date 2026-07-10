# AMD Developer Hackathon — Track 1: Token-Efficient Routing Agent

A routing agent that answers a batch of mixed-category tasks through the
Fireworks AI API while minimizing **total tokens** (prompt + completion).
It reads `/input/tasks.json`, classifies each task with pure-Python
heuristics (zero LLM calls), routes it with a tight per-category budget, and
always writes a complete `/output/results.json`.

## How it works

```
tasks.json → classifier → router → Fireworks API → validator → results.json
             (regex,      (per-cat   (kimi-k2p7-code, (pure       (every task_id,
              0 tokens)    caps +     reasoning        Python,     never empty)
                           prompts)   disabled)        0 tokens)
```

1. **Classifier** (`classifier.py`) — ordered regex/keyword rules bucket each
   prompt into `factual, math, sentiment, summarization, ner, code_debug,
   logic, code_gen` (fallback `general`). No API calls.
2. **Router** (`router.py`) — one config dict per category: model,
   `max_tokens` cap sized from measured output distributions, and a terse
   instruction. All categories route to **kimi-k2p7-code**, which won our
   5-model bakeoff on both accuracy and tokens.
3. **Reasoning disabled** — every allowed model emits hidden reasoning that
   is billed inside `completion_tokens` (up to 127 tokens for a one-word
   answer). Every call sends `reasoning_effort: "none"`, the only disable
   switch that works across all five model families. If the serving proxy
   rejects that parameter with HTTP 400, the client retries once without it
   and drops it for the rest of the run (graceful degradation to thinking-ON).
4. **Validator** (`validator.py`) — pure Python, zero LLM calls: NER answers
   must parse as JSON, Python code must pass `ast.parse`, math needs a final
   number, summaries must respect stated word/sentence/bullet limits, and
   empty or truncated (`finish_reason=length`) answers fail.
5. **Single retry** — a validation failure triggers exactly one retry with
   thinking ON and a generous cap (a tight cap + thinking would let the
   reasoning consume the whole budget and return empty content).
6. **Model fallback chain** — if a routed model is undeployed (HTTP 404), the
   call walks the remaining `ALLOWED_MODELS` in env order instead of failing.
7. **Output guard** — every input `task_id` appears in `results.json` with a
   non-empty string answer, in input order, no matter what failed upstream.
   Exit code is 0 unless input is unreadable or output unwritable.

Model IDs are never hardcoded: the router stores substrings (e.g.
`"kimi-k2p7"`) resolved against `ALLOWED_MODELS` at runtime, and the only
HTTP call site builds its URL from `FIREWORKS_BASE_URL`.

## Configuration (runtime environment variables)

| variable | purpose |
|---|---|
| `FIREWORKS_API_KEY` | API key (required) |
| `FIREWORKS_BASE_URL` | OpenAI-compatible base URL, e.g. `https://api.fireworks.ai/inference/v1` |
| `ALLOWED_MODELS` | comma-separated model IDs the agent may call |
| `INPUT_PATH` / `OUTPUT_PATH` | optional overrides (default `/input/tasks.json`, `/output/results.json`) |
| `MAX_WORKERS` | optional thread pool size (default 8) |

For local development only, a `.env` file in the working directory is read as
a fallback for unset variables. It is never baked into the Docker image.

## Local run

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your key
INPUT_PATH=tests/tasks.json OUTPUT_PATH=results.json python3 main.py
```

Every API call's model and token usage is appended to `tokens.log` (JSONL).

## Docker

```bash
# build (linux/amd64 required by the judging platform)
docker buildx build --platform linux/amd64 -t karthikrshetty5/amd-agent:latest .

# run exactly like the harness
docker run --rm \
  -v "$PWD/tests:/input:ro" -v "$PWD/output:/output" \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=... \
  -e ALLOWED_MODELS=... \
  karthikrshetty5/amd-agent:latest
```

The image contains only the five pipeline files and pinned dependencies
(~48 MB compressed). No secrets, tests, or dev tooling are copied in.

## Dev tooling (not shipped in the image)

- `eval.py` — runs the exact production pipeline on `tests/tasks.json`
  (40 tasks, 5 per category) and grades answers with an LLM judge;
  reports accuracy, per-category/per-model tokens, retries. `--sample N`
  runs a stratified subset.
- `bakeoff.py` — compares every allowed model on the test set and probes
  reasoning behavior (hidden-reasoning detection, disable-switch testing).

Current dev-set results: **40/40 judge accuracy at ~6.5–7.2k total tokens**
(~165 tokens/task), 15–23 s wall clock for 40 tasks with 8 workers.
