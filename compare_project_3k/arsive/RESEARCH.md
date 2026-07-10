# Token-Minimization Research Notes (evidence → decision)

Deep research pass (2023–2026 literature, cross-verified). Every finding is
tied to a concrete decision for our architecture. Score = accuracy gate +
fewest total tokens; work done inside the container is free.

## Verified findings and their decisions

### 1. PAL/PoT is the highest-leverage technique (CONFIRMED 3-0, arxiv 2211.10435)
Offloading math to Python beats PaLM-540B CoT by +6.4pt on GSM8K (same model).
→ **ALREADY DOING IT** (mathPAL). Decision: extend EvalExpr (sqrt/abs/percent)
so more problems fit PAL and fewer fall back to direct solving.

### 2. Extractive Lead-N summaries beat graph methods (CONFIRMED 3-0, arxiv 2512.08764)
Naive "first sentence" (Lead-1) beats most small abstractive LLMs AND all graph
methods (TextRank/LexRank) on ROUGE/BERTScore. **Don't write TextRank/LexRank —
wasted effort.**
→ Decision: for summarization, do **local extractive sentence selection** before
sending the passage to the API (cuts input tokens ~10x), then let a small model
fix the format. NOT pure-extractive (our summaries carry strict format
constraints — "20 words", "one sentence", "two viewpoints" — which Lead-N can't
honor). But see the **major risk in #3**.

### 3. ⚠️ BIGGEST RISK: will an LLM judge accept extractive/rule-based output? (NO EVIDENCE)
All classical-NLP evidence is measured with ROUGE/BERTScore/F1 — NONE with an
LLM judge.
→ Decision: never hard-wire a category to zero tokens blindly. Once the key
arrives, A/B against the real judge prompt is mandatory. Extractive summaries
and lexicon sentiment do not enter production before passing that gate.

### 4. Prompt compression: extractive chunk selection > LLMLingua token pruning (CONFIRMED 3-0)
Berkeley study: extractive selection loses almost nothing up to 10x;
LongLLMLingua is frequently the WORST. LLMLingua-2's "lossless" claim was
REFUTED (0-3).
→ Decision: local extractive selection for long-passage tasks (summaries, maybe
long factual). Don't use LLMLingua. Compress only the passage, NEVER the
instruction or the question.

### 5. Compression collapses on reasoning tasks (CONFIRMED 3-0)
GSM8K at 20x loses only −1.5pt BUT BBH at 7x loses −13.2pt. High ratios kill
logic puzzles.
→ Decision: don't compress logic/math passages aggressively; keep ratios
conservative.

### 6. ASKING for justification in sentiment REDUCES accuracy (fetch, arxiv 2406.11980)
Adding explanations pushed ChatGPT's "neutral" label rate from 19% to 54% —
burning tokens AND hurting accuracy.
→ Decision: **label-only sentiment** (drop the justification) — a token win and
a likely accuracy win. Our current prompt asks for a "one-sentence
justification" = fix it. (Revert if the judge wants a rationale — via A/B.)

### 7. Terse answers are safe for factual/math (CONFIRMED 3-0, arxiv 2410.02736)
On factual gates the quality gap dominates; style barely moves it; verbosity
bias is mostly neutral on single-answer pass/fail gates. → Decision:
answer-only/no-preamble prompts are correct, keep them. Do NOT pad to exploit
verbosity bias (net negative).

### 8. Disabling thinking is the highest-confidence completion saver — BUT measure it (PARTIAL)
The claim that constrained decoding is "free" was REFUTED (0-3). Some instruct
models emit thousands of tokens even with thinking off. → Decision: VERIFY the
Gemma/MoE thinking-off knob against live Fireworks and measure completion
lengths. Don't assume.

### 9. Chain of Draft: ultra-short reasoning where logic is genuinely needed (fetch, strong)
CoD cuts CoT completions by 68–92% while holding accuracy (GSM8K ~200→~40
tokens).
→ Decision: where logic/math truly needs reasoning, use a "draft"-style prompt
("each step ≤5 words") instead of full CoT. Completion-token win.

### 10. Token-metric routing: same tokenizer = same cost (LOW confidence, structural inference)
The three Gemmas share one tokenizer → 31b is NOT more expensive than a4b (same
tokens per string), just more accurate = fewer retries. → Decision: make
**gemma-4-31b-it** the default escalation (fewer retries, token-neutral). One
check: is a4b less chatty (MoE)? Compare completion lengths live.
Self-consistency is replaced by free local verification — already our design.

## Refuted claims (do NOT do)
- LLMLingua-2 "lossless compression" — refuted 0-3
- Constrained/structured decoding is "accuracy-free" — refuted 0-3 (validate against the gate)
- The "aggressive compression inflates total tokens" paradox — 0-3 (doesn't hold at mild ratios)

## Actionable priorities (by key availability)

**Doable now (build without the API, behind flags, A/B-ready):**
- Sentiment label-only mode (env flag)
- Local extractive sentence pre-selection for summaries (shorten the passage, then hand to the model)
- Chain-of-Draft prompts (logic/math)
- EvalExpr expansion (PAL coverage)

**MANDATORY A/B once the key arrives (against the real judge):**
- Do extractive summaries / lexicon sentiment pass the judge (#3 — most critical)
- Does thinking-off actually shorten completions (#8)
- 31b vs a4b completion lengths (#10)
- Gate-pass impact of every terse-output change
