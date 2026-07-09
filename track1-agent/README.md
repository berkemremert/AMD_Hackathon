# Fine-Tune a Query Router for the AMD Developer Hackathon

A fine-tuned local classifier that routes queries to Fireworks-hosted models (cheap vs. escalation tier), built and evaluated for AMD Developer Hackathon: ACT II's Track 1 ("Hybrid Token-Efficient Routing Agent"). Companion code for the lablab.ai tutorial: [Fine-Tune a Query Router to Cut LLM Costs for AI Hackathons](https://lablab.ai/ai-tutorials/fine-tune-llm-query-router-amd).

The router itself never calls Fireworks, it's a local forward pass, so it costs zero tokens under the hackathon's scoring rules. Only the actual answer-generating call goes through Fireworks.

> **⚠️ Disclaimer — allowed models:** per the official Participant FAQ for AMD Developer Hackathon: ACT II, the models you may use on Fireworks AI for this hackathon are the **MiniMax** and **Kimi K** series. The model IDs used throughout this repo (`minimax-m3`, `kimi-k2p7-code`, and the `glm-5p2` judge) were the serverless models available during development and are stand-ins to demonstrate the methodology. All model IDs live in `.env`, so swap in the allowed models from the hackathon's `ALLOWED_MODELS` list before submitting — no code changes needed.

## What's in here

| File | Purpose |
|---|---|
| `fireworks_client.py` | Thin wrapper around the Fireworks chat completions API |
| `code_exec.py` | Sandboxed test execution for grading generated code (code_generation category) |
| `data/generate_queries.py` | Generates the base query set across 8 capability categories, each with a verified ground truth |
| `data/generate_adversarial.py` | Adds a harder, adversarially-designed batch of queries |
| `data/label_dataset.py` | Empirically labels each query "easy"/"hard" by actually calling both models and grading the results |
| `router/train_router.py` | Fine-tunes DistilBERT as a binary easy/hard classifier |
| `router/infer_router.py` | Loads the fine-tuned router and predicts easy/hard for a prompt (local, zero tokens) |
| `baseline_router.py` | Prompt-based routing baseline (asks an LLM to classify before answering) |
| `agent.py` | Container entrypoint matching the hackathon's `/input/tasks.json` → `/output/results.json` contract |
| `evaluate.py` | Compares all 4 routing approaches on the same held-out test set |
| `Dockerfile` | Builds the submission container |
| `demo_app.py` | Streamlit UI that shows the routing decision live, for recording a walkthrough |

## Setup

1. Clone the repo and create a virtual environment:

   ```bash
   git clone <this-repo-url>
   cd fine-tune-llm-query-router-amd
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your Fireworks API key:

   ```bash
   cp .env.example .env
   ```

   ```
   FIREWORKS_API_KEY=your_key_here
   FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
   MODEL_CHEAP=accounts/fireworks/models/minimax-m3
   MODEL_EXPENSIVE=accounts/fireworks/models/kimi-k2p7-code
   MODEL_JUDGE=accounts/fireworks/models/glm-5p2
   ```

   Check which models are actually callable on your account before running anything:

   ```bash
   curl -s https://api.fireworks.ai/inference/v1/models \
     -H "Authorization: Bearer $FIREWORKS_API_KEY"
   ```

   Swap `MODEL_CHEAP`/`MODEL_EXPENSIVE` for whatever `ALLOWED_MODELS` actually lists on hackathon launch day.

## Rebuilding the dataset (optional, costs real API calls)

`data/labeled_dataset.json` is already committed with real, previously-collected results, so you don't need to regenerate it to run the router or evaluation. To rebuild it from scratch:

```bash
python3 data/generate_queries.py       # writes data/queries_raw.json
python3 data/generate_adversarial.py   # appends a harder batch
python3 data/label_dataset.py          # calls both models + a judge, writes labeled_dataset.json
```

`label_dataset.py` is resumable: it skips any query ID already present in the output file, so you can stop and restart it without re-labeling everything.

## Fine-tuning the router

```bash
python3 router/train_router.py
```

Trains DistilBERT (66M params) on `data/labeled_dataset.json`. Runs on Apple Silicon via PyTorch's MPS backend automatically if available, falls back to CPU otherwise. On AMD Developer Cloud, this same script runs unchanged on a ROCm-backed MI300X instance since PyTorch handles the backend selection.

Saves the fine-tuned weights to `router/checkpoints/router-distilbert/` (gitignored, regenerate locally, don't commit model weights to this repo).

## Demo UI

A small Streamlit app that shows the routing decision live, useful for recording a walkthrough: type a query, watch the fine-tuned router decide locally, watch the prompt-based baseline pay for the same decision, then see the real Fireworks answer and both approaches' token counts.

```bash
pip install -r requirements-demo.txt
streamlit run demo_app.py
```

Not part of the hackathon submission, the container only runs `agent.py`.

## Running the evaluation

```bash
python3 evaluate.py
```

Compares always-cheap, always-expensive, the prompt-based baseline, and the fine-tuned router on the same held-out test split, reporting total tokens and accuracy for each. Writes `evaluation_results.json`.

## Testing the container locally

```bash
docker build -t router-agent .

mkdir -p /tmp/output
docker run --rm \
  -v "$(pwd)/sample_input.json:/input/tasks.json:ro" \
  -v /tmp/output:/output \
  -e FIREWORKS_API_KEY="$FIREWORKS_API_KEY" \
  -e FIREWORKS_BASE_URL="$FIREWORKS_BASE_URL" \
  -e MODEL_CHEAP="$MODEL_CHEAP" \
  -e MODEL_EXPENSIVE="$MODEL_EXPENSIVE" \
  router-agent

cat /tmp/output/results.json
```

`ROUTER_MODE` (env var, default `finetuned`) selects the routing strategy: `finetuned`, `baseline`, `always-cheap`, or `always-expensive`.

## A note on the data

The labeled dataset is heavily skewed (80 easy / 3 hard out of 83 queries), even after a dedicated adversarial batch designed to stress-test the cheap model. That's a real, measured finding, not a generation bug: for this specific pair of Fireworks models, there's barely an accuracy gap to route around. See the tutorial for what that means for the router's training result and for picking your own model tiers.

## License

MIT
