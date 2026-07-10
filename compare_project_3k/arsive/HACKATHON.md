# AMD Developer Hackathon: ACT II — Track 1 Fact File

Single source of truth: rules, scoring, constraints, and organizer clarifications.
Last updated: 8 July 2026 (current official Participant Guide + Steve/lablab announcement).

## At a glance

- **Track 1:** "Hybrid Token-Efficient Routing Agent" / "General-Purpose AI Agent"
- **Deadline:** 11 July 2026, 18:00 CEST (Friday)
- **Prizes:** 1st $2,500 · 2nd $1,500 · 3rd $1,000 + Track 1 side prize: **$1,000 "Best Use of Gemma via Fireworks"**
- **Leaderboard:** https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live?track=1#amd-leaderboard

## Scoring

1. **Accuracy gate:** an LLM judge scores every answer against the "expected intent". Submissions below the threshold don't enter the leaderboard. The threshold is hidden.
2. **Token ranking:** gate-passing submissions are ranked ascending by **total tokens** (prompt + completion) recorded by the judging proxy. Fewer tokens = higher rank. (Raw token count, not price.)

### ⚠️ Tokens (score) ≠ dollars (credits) — don't conflate
| | Measures | Depends on | Affects |
|---|---|---|---|
| **Score/rank** | total **tokens** | how many tokens you send and receive | your leaderboard position |
| **Credit cost** | **$** spent | per-model $/token | your $50 dev budget only |

Model choice is **~neutral for the score** (the three Gemmas share one tokenizer → same string = same tokens; the difference is only chattiness + retry count). Model choice **does differ in dollars** (Kimi ~$0.95/$4.00, MiniMax ~$0.30/$1.20, Gemma cheap per 1M) → that affects dev credits only, not the score.

## ⚠️⚠️ RULES REVERSAL (8 Jul, OFFICIAL Participant Guide + Steve/lablab announcement)

The 7 Jul Discord statement "local models cannot be bundled" was **officially reversed.**
The current guide (Rules section) — binding text:

> **"Local models and tokens used locally count as zero for the final score;
> all Fireworks API calls must go through FIREWORKS_BASE_URL; local model
> inference inside the container is permitted and counts toward accuracy,
> but not toward the token score."**

- **A local model is a valid scoring strategy.** A correct local answer counts
  fully toward accuracy at 0 Fireworks tokens — "the best possible outcome for
  ranking" (Steve).
- The `ZERO_API_CALLS` marker is not a failure but an explicitly **valid strategy** (guide, p.6).
- **Grading environment: 4 GB RAM, 2 vCPU** (CPU-only). Guide sizing: 2B–3B
  4-bit quantized models are safe; 7B 4-bit fills the entire RAM budget
  (leaving no room for agent code).
- **No Ollama/runtime pre-installed** — model weights must be baked directly
  into the image (within the 10 GB compressed limit).
- The harness injects its own `FIREWORKS_API_KEY` — do not bundle your own key
  (.env is for local dev only).
- Still valid from 7 Jul: conditional model selection is allowed; plain-code
  solvers are legitimate ("plain code = zero tokens"); all Fireworks calls go
  through `FIREWORKS_BASE_URL`.

**Current winning formula:** solve what plain code can solve in code (0 tokens)
→ answer the rest with a **small local model baked into the image** (0 tokens,
counts toward accuracy) → route only tasks whose local answer provably cannot
pass the gate (or is too risky) to Fireworks with minimal tokens. Theoretical
best score: a gate-passing `ZERO_API_CALLS` run.

### Failure statuses (guide)

`PULL_ERROR` (missing amd64 manifest) · `RUNTIME_ERROR` (non-zero exit) ·
`TIMEOUT` (>10 min) · `OUTPUT_MISSING` · `INVALID_RESULTS_SCHEMA` ·
`MODEL_VIOLATION` (non-listed Fireworks model) · `IMAGE_TOO_LARGE` ·
`ACCURACY_GATE_FAILED`. Leaderboard refreshes in ~5 min (currently shows rank only).

### Official practice tasks (NOT the real set — guide, p.3)

8 examples: two-part factual (capital + body of water), mixed percent+absolute
math, contrastive sentiment, one-sentence summary, NER (person + company +
location + relative date), `get_max(nums): return nums[0]` debugging, a
**single-domain pet puzzle** (Sam/Jo/Lee — covered by our single-assignment
solver), second-largest-with-duplicates codegen. → Bundled in `eval/practice.json`;
validate container I/O against these instead of burning a submission slot.

## ALLOWED_MODELS (Track 1, published on launch day)

| Model | Character | Our use |
|---|---|---|
| `gemma-4-26b-a4b-it` | MoE 25.2B/3.8B active, thinking can be disabled | Primary (cheap/terse) |
| `gemma-4-31b-it` | Dense 30.7B, thinking can be disabled | Hard tasks |
| `gemma-4-31b-it-nvfp4` | FP4-quantized 31B | Backup |
| `kimi-k2p7-code` | 1T code model, reasoning-heavy | Code last resort (thinking tokens are scored!) |
| `minimax-m3` | 428B MoE, thinking toggle | Avoid if possible |

## Container contract

- Input: `/input/tasks.json` → `[{"task_id","prompt"}]`
- Output: `/output/results.json` → `[{"task_id","answer"}]` (valid JSON required; malformed = 0 points)
- Env (injected by the harness, hardcoding forbidden): `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (all calls through here; bypassing calls are unrecorded), `ALLOWED_MODELS` (comma-separated)
- Exit 0 on success; non-zero on failure

## Limits and rules

- Image: public registry, **linux/amd64 manifest** required, compressed ≤ 10 GB
- Startup: ≤ 60 s · Total runtime: ≤ 10 min · Per-request response: ≤ 30 s
- Answers in English
- No hardcoding/caching answers — evaluation uses unseen prompt variants
- Submissions: 10 per hour per team
- 8 categories: factual, math, sentiment, summarization, NER, code debugging, logic, code generation

## ⚠️ Submission mechanics (8 Jul organizer clarification) — STRATEGY-CRITICAL

- **You can update your submission any time; the leaderboard refreshes ~every 5 min.** Iterating is expected.
- Rate limit: **10 submissions / hour / team.**
- **ONLY the CURRENT (latest) submitted Docker image is ranked — NOT your best-ever.** There is no "keep best" safety net. Do NOT submit multiple images hoping the best sticks; the single image live in your submission at any moment is the one scored.
- **Consequence:** the LAST image before the deadline (11 Jul 18:00 CEST) is your final score. An experimental image that regresses or fails the accuracy gate *replaces* your good rank until you re-submit a better one.
- **Ladder rule:** iterate freely, but keep a **champion tag** (best-verified image) and always be able to re-submit it in one command. Never leave a worse/failing image as current — especially in the final hour. Endgame: submit the champion last and stop.

## Dev workflow (organizer recommendation)

- **Develop and test against a local model; preserve credits.** "Keep your
  development and testing off the Fireworks API unless you want to buy more
  credits." Switch to Fireworks once the solution meets the bar (accuracy
  first, then fewest tokens).
- In practice: the same binary points at a local llama-server via env
  (`FIREWORKS_BASE_URL=localhost:8080`, `ALLOWED_MODELS=local`). The client is
  endpoint-agnostic. Dev/A-B runs on free local Gemma; only final validation
  hits Fireworks.
- Caveat: local small Gemma < real Fireworks Gemma 31B. Local tests are
  reliable for routing/format/token counts; exact accuracy + exact tokens need
  a small real-Fireworks pass.

## Resources / access

- Fireworks: $50 hackathon credits (+$50 for new ADP members) — Gemma models on app.fireworks.ai; the organizers quoted a Gemma 4 E4B deployment at $7/hour (for dev/experiments)
- AMD AI Notebooks: team-2678, 4 GPU-hours/day (ROCm/vLLM or Unsloth+llama.cpp images) — no longer needed for the Track 1 submission itself; an x86 test bench for perf validation
- Team: 2 members (me@omerduran.dev + amd.shelter597@passmail.net)

## Timeline / status notes

- 7 Jul: ALLOWED_MODELS learned; local-first architecture built and measured at 90–97% local accuracy on our evals; then the organizer statement banned bundled local models → **pivot to a Fireworks-only architecture** (task list #11). The leaderboard was still empty — nobody had a scored submission before the rules settled.
- 8 Jul: The official guide was updated — **the local-model ban was reversed** (section above). The pre-pivot local-first architecture (preserved in git history) is the winning strategy again; resized for 4 GB/2 vCPU (E2B instead of E4B). Track 1 scoring pipeline is live. 7 rival repos analyzed (see memory/rival notes): none ship proof-gated plain-code answers; the most serious rival is frugal-router (local GGUF brain + draft-confirm) — the new rule legitimizes their architecture too.
- 8 Jul (cont.): **Re-pivot complete** — the local tier is back (E2B Q4_K_XL 2.97 GB; the E4B file was eliminated at 4.8 GB). Host e2e: Fireworks calls tasks 59→23, hard 20→7. Two measured local failure modes closed (logic rambles → escalate; nuanced sentiment → strong model). The official practice set is in eval; a single-domain solver covers practice-07 for 0 tokens. Remaining: Docker 4g/2cpu functional validation + GHCR push (waiting on the GitHub repo) + the submission ladder.
- 8 Jul (ladder log): **Sub #1 (full local, all categories) → ACCURACY_GATE_FAILED.** Hardened in 915401f: 429/5xx backoff (was missing entirely — a rate-limit burst became fallback answers), thinking-drain rescue (empty/leaked content → one 4×-cap retry), Dockerfile DEFAULT_* build args + build-variant workflow for baked rungs. **Rung 1 = `:rung1-safe-local`** (local ONLY for proof-gated categories math/code_gen/code_debug; all free-text categories go to strong remote) — promoted to `:champion`, awaiting re-submit. Decision tree: pass → re-add local categories one at a time; fail → rung 2 = LOCAL=false with generous caps (if THAT fails, the issue is remote answer style, not the local tier).
