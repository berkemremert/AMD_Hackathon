# Performance Optimization Log

**Protocol:** every change is applied in isolation and measured on the same
benchmark. Decision rule: NO clear improvement in wall time or the call/token
metric → the change is reverted. Accuracy proxies (layer distribution,
unproven count) must not regress.

**Benchmarks:**
- `B64`: eval/paraphrased.json, host (Metal). For code-level changes —
  a reduction in call count is hardware-independent.
- `D24`: eval/hard.json, Docker container (CPU-only). For CPU-sensitive
  changes (server flags, cache) — the closest environment to the judging VM.

**Metrics:** wall time · local call count · local completion tokens ·
layer distribution (deterministic/local/unproven).

| # | Change | Benchmark | Time | Local calls | Compl. tokens | Layers (d/l/u) | Decision |
|---|---|---|---|---|---|---|---|
| 0 | Baseline (222b8b8) | B64 | 6m24s | 191 | 18578 | 4/53/7 | reference |
| 1 | Batched classification (chunks of 16) | B64 | 8m41s | 142 | 20253 | 4/53/7 | **KEEP** — calls −49, category hits 57→59; the time increase isn't from batching but from 2 tasks routing to logic (3×900-tok pipeline) in this run. Lesson: verification token volume dominates wall time, not call count → step 2 |
| 2 | Answer-only verification samples (40/80 tok) | B64 | 4m33s | 142 | 9661 | 4/46/14 | **REVERT** — time −48% but unproven 7→14: CoT-less sampling is too weak, false disagreements → ~7 needless escalations live (scored tokens!). Buying local time with scored tokens is a bad trade |
| 2b | Short-CoT samples (150/250 tok) | B64 | 5m33s | 143 | 12376 | 4/48/12 | **PARTIAL** — unproven still 12. Instead of a static pick, step 3: dynamic mode by time pressure (Full→Brief→Off) |
| 3 | Pacer: dynamic verification mode (throughput projection; Full→Brief→Off) | B64 | 6m59s | 142 | 20098 | 4/50/9(+1 brief,1 off) | **KEEP** — stays in ModeFull on a normal budget: token-optimal verification (unproven 9 ≈ baseline). Brief code engages only under pressure |
| 3b | Pressure test: TOTAL_BUDGET=3m (under half of normal) | B64@3m | 3m00s | 68→79 | 2997→3805 | mostly mode=off | **KEEP** — 64/64 answered, valid JSON. With a generation cap (120 tok) in ModeOff, 'Unable' dropped 19→8 (the remaining 8 are the physical floor of a 3-minute run). Graceful degradation works |

**Step 2 lesson (permanent):** because score = tokens, the default is always
full-CoT verification; short sampling is legitimate only under time pressure.
A static speed optimization that increases scored tokens is rejected.

## Free logic solvers (2026-07-07, post-pivot)

For rival parity, `SolveOrdering` (transitive ordering, topological sort) +
`SolveSyllogism` (syllogisms, reachability) were added — `internal/solve/logic.go`.
They run independent of category (strict self-gate → rescues a misclassified
puzzle, never fires on non-matching text).

| Set | Logic solved in code | Note |
|---|---|---|
| tasks.json | logic-3 (Fay), logic-6 (Yes) | logic-6 was classified as factual; the solver rescued it → 2 tasks at **0 tokens** |
| hard.json | 0 | lh-1/2/3 (zebra/knights/position-offset) safely deferred |
| paraphrased.json | 0 | different phrasing ("Name the winner", "does it follow") didn't match the patterns → safely deferred |

**Decision: KEEP.** Reduces canonically-phrased ordering/syllogism tasks to
0 tokens and never gives a WRONG code answer to an out-of-scope task (golden
rule preserved). Coverage is phrasing-dependent (defers on paraphrases) but
the risk is zero. Real value depends on the hidden judge's phrasing.

## Batching (2026-07-07, toggled — `BATCH_SIZE`)

Pack sentiment+factual tasks (single line, ≤300 chars) into one call → the
system prompt is paid once. `internal/router/batch.go` + a main.go pre-pass.
Free-solve runs first (invariant); a parse failure falls back to per-task calls.

Mock A/B (tasks.json; the tokenizer isn't realistic — call count is the valid
metric, token values are rough):

| Mode | Fireworks calls | Tokens (mock) |
|---|---|---|
| BATCH_SIZE=0 | 60 | 4071 |
| BATCH_SIZE=8 | **46** (−14) | 3781 |

logic-6 (classified factual) is byte-identical in both modes → the free-solve
invariant held; empty answers 0.

**Decision: KEEP THE CODE, DEFAULT OFF.** The mechanism is proven (calls −23%,
safe fallback, invariant), but real accuracy in batch mode (context bleed) and
real tokens can only be measured on live Fireworks. The mock serves canned
answers, so accuracy is unmeasurable. `BATCH_SIZE` stays a ladder knob; if
live shows tokens↓ with accuracy held, the default becomes 8.

## Stop sequences (2026-07-07, toggled — `STOP_SEQ`)

Per-category `\n\n` stop → trailing paragraphs/filler after the answer are cut
(completion tokens). sentiment/factual/summarize/ner → `\n\n`; math/logic/code →
none (they contain newlines); batch → none (the list separates by `\n`);
PAL → `\n` (the expression is one line). `internal/router` + `STOP_SEQ` config.

The mock ignores stops → the token saving is INVISIBLE on mock. Regression
check: STOP_SEQ on/off answers are byte-identical (0 diff), empty answers 0,
errors 0 → the stop parameter breaks nothing. Unit test: never a bare `\n`
stop for NER/code (truncation guard).

**Decision: KEEP THE CODE, DEFAULT OFF.** `\n\n` is conservative (preserves
single-paragraph answers), but the real completion-token reduction and any
truncation-accuracy effect can only be measured live. Ladder knob; if live
shows tokens↓ with accuracy held, the default flips on.

## Experimental component set (2026-07-07 — 7 toggled components)

Each is a separate `Options`/env toggle; decision rule: **provable-offline
components ship on, live-judge/tokenizer-dependent ones ship off** (code stays,
A/B'd in the first live round).

### PUZZLE_SOLVERS — brute-force puzzle solvers (default: ON)

`SolveKnights` (2^n truth assignments), `SolveZebra` ((n!)² two-attribute
assignment; query-cell uniqueness suffices — even when the full grid stays
ambiguous), `SolvePositions` (n! permutations; offsets/adjacency — the shapes
the ordering solver deliberately defers on). Strict self-gate: an unparsed
clue sentence or multiple solutions → defer.

| Set | Effect |
|---|---|
| hard.json | lh-1 (zebra) + lh-2 (knights) + lh-3 (positions, rescued despite factual misclassification) → **0 tokens**; calls 24→21 |
| tasks.json | no change (60 calls byte-identical) — no false grabs |
| paraphrased.json | no change — safely defers |

All three answers verified by hand. **Decision: KEEP, ON** (proof-safe; the
most expensive logic tasks became free).

### MUTATION_REPAIR — single-edit debug repair (default: ON)

Proof rule: the original code FAILS the asserts derived from the prompt's own
examples + exactly ONE mutant PASSES → 0-token answer; anything else (no
asserts / original passes / multiple passing mutants / out of time) → defer.
Unit tests: `range(1,n)→range(1,n+1)` and `a-b→a+b` repaired; three defer
scenarios verified. Eval debug tasks carry no examples → all defer (coverage 0,
risk 0 — the mechanism is unit-proven; value depends on the hidden set carrying
examples). **Decision: KEEP, ON** (proof-gated: a wrong answer is structurally
impossible).

### SOLUTION_LIB — proven solution library (default: ON)

12 classics (fibonacci ×2 conventions, palindrome ×2, reverse, prime, gcd,
anagram, brackets...) rendered with the requested function name and NEVER
answered without passing the prompt's own examples. tasks.json code-4
(60→59 calls) + hard.json ch-2 (21→20 calls) from the library, test-passed.
Example-less / foreign-language / example-contradicting tasks defer
(unit-tested). **Decision: KEEP, ON.**

### DEDUP — task deduplication (default: ON)

Normalized (lowercase+whitespace) exact duplicates copy the representative's
answer. Synthetic 6-task set (3 dups): 2 calls, duplicate answers identical
and non-empty. The eval sets carry no duplicates → no effect there, no risk.
**Decision: KEEP, ON.**

### PROMPT_COMPRESS — input compression (default: 0 = OFF)

Level 1 strips politeness/boilerplate, level 2 adds extractive sentence
selection for long summary passages (instruction + lead always preserved;
degenerate output falls back to the original). Unit-tested; **judge tolerance
is a live-only measurement** → waits disabled.

### MERGE_SYSTEM — chat template shaving (default: OFF)

The system message folds into the user message → per-message role-scaffolding
tokens are trimmed. Unit test + mock regression clean (the mock's canned
matching keyed on the system message; fixed to also match the user message in
merge mode — a mock artifact; real endpoints read instructions regardless of
role). The gain shows up only on the live tokenizer → off.

### GRAMMAR — constrained decoding (default: OFF)

GBNF on sentiment (`response_format {type: grammar}`): filler-token generation
is impossible by construction. The field is written to the body only when set
(proven via httptest); one unconstrained retry on error → losing an answer is
impossible. The mock ignores the field → the effect is measured live → off.

**Combination smoke test:** PROMPT_COMPRESS=2 + MERGE_SYSTEM=1 + GRAMMAR=1 +
STOP_SEQ=1 + BATCH_SIZE=8, tasks.json → 45 calls, 64/64 non-empty answers,
0 errors.

### components.json — targeted component probe set (2026-07-07)

22 tasks: one solver-hit plus one deliberate-defer case per component
(with `expected-components.json`). Mock run — behavior matrix 22/22 by design:

| Component | Fired (0 tokens) | Correctly deferred |
|---|---|---|
| Knights | kk-1, kk-2 (both/self-ref), kk-3 (at-least-one) | kk-4 ("is not a knight" outside the grammar) |
| Zebra | zb-1 (query cell unique while the grid stays partly ambiguous) | zb-2 (three domains) |
| Positions | pos-1 (ordinal+offset+adjacent; rescued despite factual misclassification), pos-2 | pos-3 (negation "was not last") |
| Mutation repair | mr-1 (range upper bound), mr-2 (`<`→`>` direction flip) | mr-3 (multi-token fix) |
| Library | sl-1 gcd, sl-2 vowels, sl-3 digit-sum | sl-4 (non-standard fib convention — the examples rejected the variants) |
| Dedup | dd-2 (normalized copy of dd-1, 0 extra calls) | — |

11/22 tasks at 0 tokens; 10 calls total. The content of all 11 free answers
verified against expected. Two safe extensions enabled this set:
`DeriveAsserts` now understands "should return / must be" and
"returns X **instead of** Y" (expectation = Y; the observed buggy value never
becomes an assert — regression-tested), and the mutation table gained direction
flips (`<`↔`>`). Regression: tasks.json 59, hard.json 20 calls — unchanged.

**Live measurement queue:** `BATCH_SIZE`, `STOP_SEQ`, `PROMPT_COMPRESS`,
`MERGE_SYSTEM`, `GRAMMAR` — all default off because their accuracy is
unmeasurable on mock; A/B'd one by one in the first live Fireworks round.
(`REASONING_EFFORT=low` defaults ON: fewer thinking tokens is directly score,
and rejected knobs get a plain retry — structurally zero risk.)

## Re-pivot: the local tier (2026-07-08, after the rules reversal)

### Model sizing — the 4 GB RAM / 2 vCPU grading box

| Candidate | GGUF size | Decision |
|---|---|---|
| gemma-4-E4B-it UD-Q4_K_XL | **4.8 GB** | ELIMINATED — the weight file alone exceeds the RAM budget (guide: "7B 4-bit fills the full RAM budget"; E4B is effectively in that class) |
| gemma-4-E2B-it UD-Q4_K_XL | **2.97 GB** | PRIMARY — leaves ~1 GB for weights + KV (ctx 4096, parallel 2) + agent + python3; to be confirmed with a Docker `--memory 4g` test |
| gemma-4-E2B-it Q4_0 | 2.83 GB | Fallback — if XL OOMs in the 4g test |

llama-server settings follow: `LOCAL_CTX_SIZE=4096` (modest KV),
`LOCAL_PARALLEL=2` (more slots just thrash on 2 vCPU),
`LOCAL_REQUEST_TIMEOUT=20s` (inside the 30s/request rule).

### Mock regressions (behavior byte-identical with the local tier off)

tasks.json 59 calls, components.json 10 calls — same as pre-re-pivot.
Local-tier tests: gating (nil client / category filter / pacer pressure),
local-answer-without-remote, format-failure→escalation, local PAL→Go
evaluation. practice-07 (official set) solved by the single-domain solver at
`layer=code`, 0 tokens.

### Host e2e: E2B (Metal) + mock-Fireworks — Fireworks call reduction

| Set | Before (FW-only) | After (local tier) | Local 0-token answers |
|---|---|---|---|
| tasks.json (64) | 59 calls | **23 calls** | 38 |
| hard.json (24) | 20 calls | **7 calls** | 13 |
| components.json (22) | 10 calls | **5 calls** | 5 |
| practice.json (8) | 7 calls | **3 calls** | 4 |

Time: 64 tasks ≈ 50 s (host Metal; the grading box is slower on CPU — the
pacer skips the local tier outside ModeFull). Most remaining remote calls are
example-less codegen/debug (no proof derivable → strong model by design).

### E2B accuracy spot checks and two measurement-driven fixes

**Good:** math PAL 6/6 exact; factual/NER/summary format-clean; sarcasm
sentiment (sh-1) correct; base sentiment 3/3.

**Two measured local failure modes and their fixes (KEEP):**
1. A local logic answer that never reaches its 'Answer:' line is a truncated
   reasoning dump → fails the judge. Fix: local logic ships only as a
   ≤100-char single-line conclusion; rambles go to the strong model.
   (logic-2 additionally dropped to a full 0 tokens once the knights solver
   learned 'We are both knaves' + silent-participant harvesting.)
2. Nuanced sentiment: contrastive dominant-verdict (sh-2: said Mixed, expected
   Positive-overall) and factual-report-neutral (sh-3: said Negative). Both
   are lexically marked (but/however, describes/reports) →
   `reSentimentNuance` matches skip the local tier. Cost on the base set:
   +4 remote calls (~200 tokens) — cheap insurance against 2 likely gate
   failures.

mh-1's '222' is the mock's canned PAL expression (meaningful live), not a
local failure.

## Local model bake-off: E2B vs Gemma-3-4B (2026-07-08)

Research (research-repivot-findings) suggested a stronger 3-4B model that fits
4 GB might raise Layer-1 accuracy. Tested empirically: ran all four eval sets
against **Gemma 4 E2B** (current, ~3 GB Q4_K_XL) and **Gemma 3 4B**
(~2.5 GB Q4_K_M), both real llama-server, then judged the 54 model-dependent
answers with a 3-judge majority panel (159 judge calls). Both are Gemma-family
(safe for the side prize; the local model is NOT bound by ALLOWED_MODELS).

Result `[localCorrect, localWrong, escalated]`:

| Category | E2B | Gemma-3-4B | Winner |
|---|---|---|---|
| math | [7,0,2] | [4,**5**,0] | **E2B (decisive)** |
| summarize | [11,0,0] | [6,3,2] | **E2B (decisive)** |
| ner | [7,1,0] | [6,2,0] | E2B |
| factual | [12,1,0] | [12,1,0] | tie |
| sentiment | [6,0,0] | [6,0,0] | tie |
| logic | [1,0,3] | [4,0,0] | Gemma-3-4B |
| **TOTAL** | **46 / 2 / 5** | **39 / 12 / 2** | **E2B** |

**Decisive metric = local-WRONG** (a 0-token answer that is wrong = an accuracy
-gate loss for that task): **E2B 2 vs G4B 12 (6×).** G4B keeps more tasks local
(2 escalations vs 5) but its extra local answers are disproportionately WRONG —
trading ~a few hundred tokens of savings for 10 more wrong-answer risks, a bad
trade in a token-scored-but-accuracy-gated contest.

Root cause: E2B is better CALIBRATED to our free-verification gates — it
escalates when unsure (math it can't express, logic it can't conclude);
G4B pushes confident-but-wrong answers, especially on **PAL math** (a valid
expression that mis-models the problem: 162→31.48, 1320→−1320, 3→−5) where
there is no post-hoc correctness check, only arithmetic-validity. G4B's only
edge is logic (4 local-correct vs E2B's 3 escalations) — but E2B answers those
CORRECTLY via escalation for a few hundred tokens, so the gap is cost not
accuracy, and it's dwarfed by the math regression. G4B is also 4B-dense →
slower on the 2 vCPU box (timeout risk) and sometimes adds a preamble that
breaks a one-sentence constraint (summ-7).

**Decision: KEEP E2B.** The stronger-model hypothesis is empirically REFUTED
for our pipeline: raw model strength matters less than calibration to the
verification gates. (Latent risk surfaced: the PAL path accepts any valid
arithmetic expression without checking it models the problem — E2B happens to
be reliable here, but a future hardening could cross-check two independent PAL
samples. Not pursued now.)

## First LIVE Fireworks smoke test (2026-07-08, personal key, stand-in models)

The hackathon's exact model IDs exist only behind the harness proxy; the
personal account serves other serverless models, so a reasoning model
(kimi-k2p6) stood in for all roles — which is exactly what made the test
valuable. All measured live:

1. **Reasoning tokens are billed as completion tokens.** At
   `reasoning_effort=low` the model spent **31** completion tokens producing a
   2-token answer; with a tight `max_tokens` the truncated thinking **leaked
   into the content field**. `reasoning_effort=none` disabled thinking
   entirely: completion **31 → 2 (15×)**, identical answer. → Default changed
   to `none` (plain-retry fallback covers endpoints that reject it); the
   client now parses `reasoning_content` separately so thinking is never
   mistaken for an answer.
2. **Prefix caching is real and measurable:** two identical-prefix calls with
   the `x-session-affinity` header → `cached_tokens` **0 → 54 of 62** on the
   second call. The PREFIX_CACHE design is confirmed against the live API.
3. **Official practice set end-to-end (remote-only, LOCAL=0):** 8/8 answered,
   valid schema, 5.3 s, 7 calls / **598 tokens**. Real PAL returned the exact
   expected 144 (unmeasurable on the mock); the remote tier corrected the one
   fact the local tier gets wrong (practice-01 Lake Burley Griffin),
   validating the escalation design. practice-07 stayed at 0 tokens (solver).

**Verdict: the baseline image is submission-ready.** Caveat: Pick() role
separation (Gemma-vs-Kimi) could not be exercised — the harness's own
ALLOWED_MODELS resolve that at eval time; the fallback path it uses here is
itself now live-tested.

### Docker functional validation (amd64 image, `--memory 4g --cpus 2`)

Image: **3.14 GB** uncompressed (well under the 10 GB compressed limit).
practice.json run under qemu emulation on Apple Silicon (5–10× slower than
real x86 — functional signal only):

- llama-server loaded the bundled E2B **in ~21 s** from container start
  (inside the 55 s gate) — no OOM under the 4 GB limit (weights + ctx 4096 fit).
- Emulated generation was too slow for the 20 s local timeout → every local
  attempt escalated gracefully; **8/8 answered, valid JSON, exit 0, 59.6 s
  total**. The degradation path is proven end-to-end in-container.
- practice-07 answered by the single-domain solver inside the container
  (0 tokens).

### CPU throughput proxy (host llama-server, `-ngl 0 -t 2`, no GPU)

Native CPU with 2 threads approximates the grading box far better than qemu:
tasks.json (64) finished in **2m45s** — 35 local answers (0 scored tokens),
25 Fireworks calls (2 more than the Metal run: a couple of slow locals hit
the 20 s timeout and escalated, as designed). Even if the grading box's vCPUs
are 2× slower per core, the projection (~5.5 min) stays inside the 10-minute
budget with the pacer as backstop.
