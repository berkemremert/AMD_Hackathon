# Token-Efficient Routing Agent

A general-purpose AI agent built for **Track 1 of the AMD Developer Hackathon: ACT II**. It answers tasks across eight capability domains through the Fireworks AI API, engineered to spend the **fewest tokens possible** while staying accurate.

> Built solo by **Adeel Faheem** · Team **SXR**

---

## The idea

Track 1 is scored in two stages: first an **accuracy gate** (an LLM judge must approve the answers), then a ranking by **fewest total tokens used**. So the goal isn't just to answer correctly — it's to answer correctly *while spending as little as possible*.

This agent's strategy: understand each task before answering, and give every task only as much as it needs.

1. **Classify** each incoming task into one of eight categories using lightweight pattern matching — this costs **zero tokens**.
2. **Tighten** the request per category: the leanest instruction and the smallest safe output limit for that kind of task.
3. **Call** Fireworks AI once, cleanly, preferring a Gemma model (to also compete for the Best-Use-of-Gemma challenge).

## Capability domains

Built for all eight Track 1 categories:

| # | Category | # | Category |
|---|----------|---|----------|
| 1 | Factual knowledge | 5 | Named entity recognition |
| 2 | Mathematical reasoning | 6 | Code debugging |
| 3 | Sentiment classification | 7 | Logical reasoning |
| 4 | Text summarisation | 8 | Code generation |

## How it works

```
/input/tasks.json
      |
      v
  classify        zero-token category detection (plain code)
      |
      v
  tighten         per-category instruction + output cap
      |
      v
  Fireworks AI    one call - allowed model - Gemma preferred
      |
      v
/output/results.json
```

The container reads tasks from `/input/tasks.json` and writes answers to `/output/results.json`, matching the hackathon's evaluation contract. All credentials and model IDs are read from environment variables injected at runtime.

## Engineering highlights

- **Zero-token task classification** in plain Python — no model call needed to route.
- **Per-category output caps** that trim wasted completion tokens without risking the accuracy gate.
- **Reasoning-model handling** — cleanly extracts the final answer (e.g. the code block) when a model "thinks out loud" in a separate field.
- **Automatic model fallback** across the allowed model list, so a single model outage can't break a run.
- **Fully containerized** with Docker, targeting `linux/amd64`.

## Running it

Locally (reads credentials from a `.env` file or the environment):

```bash
pip install -r requirements.txt
INPUT_PATH=practice_tasks.json OUTPUT_PATH=results.json AGENT_MODE=remote python src/main.py
```

With Docker, the way the evaluation harness runs it:

```bash
docker build -t token-router .
docker run --rm \
  -v "$PWD/input:/input" -v "$PWD/output:/output" \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/gemma-4-31b-it \
  token-router
```

Environment variables (`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`) are never hardcoded and never committed — the `.env` file is git-ignored.

## Project layout

| Path | Purpose |
|------|---------|
| `src/main.py` | Container entry point — reads `/input`, writes `/output` |
| `src/prompts.py` | Zero-token category classifier + per-category tight prompts |
| `src/models.py` | Fireworks API call, model selection, reasoning-answer extraction |
| `src/agent.py` | Pipeline orchestration |
| `src/tokens.py` | Token accounting |
| `Dockerfile` | Containerization (linux/amd64) |
| `practice_tasks.json` | Sample tasks spanning all eight categories |

## Tech

Python - Docker - Fireworks AI - Gemma

---

*AMD Developer Hackathon: ACT II — Track 1 · Team SXR*
