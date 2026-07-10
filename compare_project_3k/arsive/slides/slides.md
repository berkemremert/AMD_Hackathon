---
theme: default
colorSchema: dark
class: text-center
highlighter: shiki
lineNumbers: false
transition: slide-left
mdc: true
title: TokenRouter — the prove-or-escalate agent
info: |
  TokenRouter: a token-minimizing hybrid routing agent.
drawings:
  persist: false
---

<div class="glow w-100 h-100 bg-orange-500 -top-20 -right-20"></div>
<div class="glow w-80 h-80 bg-blue-600 -bottom-24 -left-16"></div>

<div class="kicker mb-6">AMD Developer Hackathon · ACT II · Track 1</div>

<h1 class="no-bar" style="font-size:5rem; line-height:1.05;">
<span class="num-hot">Token</span><span class="num-warm">Router</span>
</h1>

<div class="text-2xl mt-2 font-light tracking-wide">the prove-or-escalate agent</div>

<div class="pt-10 text-xl opacity-70">
It never pays for what it can prove.
</div>

---
layout: center
class: text-center centered-h
---

<div class="glow w-90 h-90 bg-orange-500 top-10 -left-30"></div>

# One agent, eight task categories.<br>Two-thirds of the time, it pays <span class="num">nothing</span>.

<div v-click class="mt-10 text-lg opacity-90 max-w-170 mx-auto leading-relaxed">

Math, logic, code, facts, sentiment, NER, summaries — TokenRouter answers them all,
but treats every API token as a **purchase that requires evidence**:
it proves an answer with free computation first, and buys one
only on a **proven** miss. Never on a hunch.

</div>

<div v-click class="card card-accent inline-block mt-10 px-8">
<div class="text-sm opacity-70 mb-1">the golden rule of its free tiers</div>
<div class="text-2xl font-bold"><span class="num">no proof → no answer → escalate</span></div>
<div class="text-sm opacity-70 mt-1">a wrong free answer is structurally impossible</div>
</div>

---

<div class="kicker">Architecture</div>

# Three tiers, one ladder

<div class="grid mt-10 items-stretch" style="grid-template-columns: 1fr auto 1fr auto 1fr; gap: 0;">

<div class="card" style="border-color: rgba(255,122,26,0.55);">
<div class="kicker mb-2">Tier 0 · 0 tokens</div>
<div class="font-bold text-lg mb-3">Plain Go code</div>
<div class="text-sm leading-relaxed opacity-90">
arithmetic AST<br>
6 puzzle solvers<br>
mutation repair<br>
proven library
</div>
</div>

<div class="flex flex-col justify-center px-3 text-center">
<div class="text-2xl num">→</div>
<div class="text-[0.65rem] opacity-60 mt-1">no<br>proof</div>
</div>

<div class="card" style="border-color: rgba(255,122,26,0.55);">
<div class="kicker mb-2">Tier 1 · 0 tokens</div>
<div class="font-bold text-lg mb-3">Bundled Gemma 4 E2B</div>
<div class="text-sm leading-relaxed opacity-90">
code runs against the<br>prompt's own tests<br>
math recomputed in Go<br>
format & length gates
</div>
</div>

<div class="flex flex-col justify-center px-3 text-center">
<div class="text-2xl num">→</div>
<div class="text-[0.65rem] opacity-60 mt-1">verified<br>miss</div>
</div>

<div class="card" style="border-color: rgba(100,116,139,0.7); background: rgba(148,163,184,0.06);">
<div class="kicker mb-2" style="color:#94a3b8;">Tier 2 · minimal spend</div>
<div class="font-bold text-lg mb-3">Fireworks API</div>
<div class="text-sm leading-relaxed opacity-90">
terse prompts · tight caps<br>
reasoning_effort: none<br>
prefix-cache affinity<br>
budget-capped retry
</div>
</div>

</div>

<div class="flex items-center justify-center gap-4 mt-8">
<div class="px-4 py-1 rounded-full text-sm font-bold" style="background:#ff7a1a; color:#14100a;">task</div>
<div class="flex-1 border-t border-dashed" style="border-color: rgba(255,255,255,0.25); max-width: 18rem;"></div>
<div class="text-sm opacity-70">any tier may answer — with <b>evidence</b></div>
<div class="flex-1 border-t border-dashed" style="border-color: rgba(255,255,255,0.25); max-width: 18rem;"></div>
<div class="px-4 py-1 rounded-full text-sm font-bold" style="background:#fbbf24; color:#14100a;">answer</div>
</div>

<div class="mt-6 text-center opacity-60 text-sm">
Each tier is cheaper than the next. Climbing requires <b>evidence</b>, not vibes.
</div>

---
layout: two-cols-header
---

<div class="kicker">Tier 0</div>

# The free floor

::left::

<div class="text-[0.92rem] pr-6">

**Deterministic solvers** — plain Go, microseconds:

- Arithmetic expression evaluator
- Knights-and-knaves — brute-force 2ⁿ truth tables
- Zebra grids — exhaustive (n!)² assignment
- Positional races, orderings, syllogisms
- **Mutation repair** — single-edit mutants of buggy code, tested against asserts derived from the prompt's own examples
- **Proven-solution library** — classics ship only after passing the prompt's own examples

</div>

::right::

<div class="text-sm opacity-70 mb-2">real trace, official practice set</div>

```text
task practice-02  layer=pal   → "144"
task practice-07  layer=code  → "Sam owns the cat."   0 tokens
task practice-05  layer=remote → NER, 4/4 entities
```

<div v-click class="card card-accent mt-5 text-sm">
Every solver <b>self-gates</b>: an unparsed clue or an ambiguous
solution means <i>defer</i>, never guess.
<br><br>
A deferral costs a few tokens.<br>A wrong answer costs the accuracy gate.
</div>

---
layout: two-cols-header
---

<div class="kicker">Tier 1</div>

# A local model you can trust

::left::

<div class="text-[0.92rem] pr-6">

A **Gemma 4 E2B** (Q4 GGUF, llama.cpp) ships *inside* the image —
sized for the 4 GB / 2 vCPU grading box.

Local inference counts toward accuracy and **zero** toward the
token score. But a small model hallucinates — so nothing ships
unverified:

- generated code → **executed** against prompt-derived tests
- math → model emits an expression, **Go recomputes it**
- everything else → format, length & refusal gates

</div>

::right::

<div v-click class="card card-accent text-sm">

**Research-driven detail:** terse "draft" prompting costs sub-3B models
16–27 accuracy points *(Chain-of-Draft, arXiv 2502.18600)*.

So the local tier gets **full chain-of-thought** — its tokens are
free — while the paid tier stays terse.

</div>

<div v-click class="card mt-4 text-sm opacity-90">
Under deadline pressure a throughput pacer skips this slow CPU tier
entirely — buying time with tokens, gracefully.
</div>

---

<div class="kicker">Tier 2</div>

# When we do pay, we never pay list price

<div class="grid grid-cols-3 gap-5 mt-8">

<div v-click class="card text-center">
<div class="text-4xl num">31 → 2</div>
<div class="mt-2 text-sm opacity-80">completion tokens for the same answer with
<code>reasoning_effort: none</code> — <b>measured live</b>; thinking tokens are billed like any others</div>
</div>

<div v-click class="card text-center">
<div class="text-4xl num">54 / 62</div>
<div class="mt-2 text-sm opacity-80">prompt tokens served from cache on the second call —
session-affinity keeps the shared prefix warm (<b>measured live</b>)</div>
</div>

<div v-click class="card text-center">
<div class="text-4xl num">1 retry</div>
<div class="mt-2 text-sm opacity-80">paid retries happen only on <b>proven</b> failure
(failed tests, broken format) and are capped by a global budget knob</div>
</div>

</div>

<div v-click class="mt-8 text-center opacity-70 text-sm">
terse category prompts · per-category <code>max_tokens</code> · PAL for math — ~20 tokens instead of a worked solution
</div>

---
layout: center
class: text-center centered-h
---

<div class="glow w-100 h-100 bg-orange-500 -top-10 right-0"></div>

<div class="kicker">Measured results</div>

<h1 class="no-bar"><span class="num" style="font-size:6.5rem; line-height:1;">59 → 23</span></h1>

<div class="text-xl mt-1">Fireworks calls on our 64-task eval — <b>−61%</b></div>

<div class="grid grid-cols-3 gap-6 mt-12 text-center">

<div class="card">
<div class="text-3xl num">~2 / 3</div>
<div class="text-sm opacity-75 mt-1">of tasks answered at<br><b>zero scored tokens</b></div>
</div>

<div class="card">
<div class="text-3xl num">20 → 7</div>
<div class="text-sm opacity-75 mt-1">calls on the deliberately<br>hard eval set (−65%)</div>
</div>

<div class="card">
<div class="text-3xl num">2m 45s</div>
<div class="text-sm opacity-75 mt-1">64 tasks on a 2-thread CPU proxy<br>(10-minute budget)</div>
</div>

</div>

<div class="mt-8 text-xs opacity-50">
measured on our own eval sets against the real local model — honestly labeled, no leaderboard claims
</div>

---
layout: two-cols-header
---

<div class="kicker">Engineering culture</div>

# We measure. Even when it kills our favorite idea.

::left::

<div class="pr-8 text-[0.92rem]">

**The bake-off:** research said *"a stronger 3–4B local model should
raise free-tier accuracy."* We tested it — 54 tasks, two real models,
a 3-judge panel per answer.

<div class="mt-4">

| | free & correct | free & **wrong** | escalated |
|---|---|---|---|
| **Gemma 4 E2B** | **46** | **2** | 5 |
| Gemma 3 4B | 39 | <span class="text-red-400 font-bold">12</span> | 2 |

</div>

</div>

::right::

<div v-click class="text-[0.92rem]">

The bigger model answered *more* — and was wrong **6× more often**.
Every wrong free answer is an accuracy-gate loss.

</div>

<div v-click class="card card-accent mt-4 text-sm">
<b>Lesson:</b> raw model strength matters less than
<b>calibration to the verification gates</b>. E2B escalates when unsure;
the 4B pushed confident-but-wrong answers.
</div>

<div v-click class="mt-4 text-xs opacity-60">
Every optimization lives in a perf journal: applied in isolation, measured,
kept only if it helped. Unproven features ship behind default-off flags.
</div>

---

<div class="kicker">Robustness</div>

# It cannot crash, stall, or emit bad JSON

<div class="grid grid-cols-2 gap-x-8 gap-y-3 mt-6 text-[0.9rem]">

<div class="card">📄 <b>Skeleton <code>results.json</code> at startup</b> — even an instant crash leaves valid, scorable JSON</div>
<div class="card">🧯 <b>Per-task panic isolation</b> — one adversarial prompt can't take down the run</div>
<div class="card">⚛️ <b>Atomic writes</b> (temp + rename) — output can never be half-written</div>
<div class="card">⏱️ <b>Throughput pacer</b> — projects the finish time, degrades gracefully under pressure</div>
<div class="card">🛑 <b>SIGTERM flush</b> — an early kill still submits everything answered so far</div>
<div class="card">🧪 <b>Fuzz-tested</b> — 200k-char prompts, control bytes, unicode floods: no panics, valid output</div>

</div>

<div v-click class="card card-accent mt-6 text-center">
The image passed the real contract under grading limits:<br>
<code>--memory 4g --cpus 2</code> → model loaded in 21 s, 8/8 answered, exit 0.
</div>

---
layout: center
class: text-center centered-h
---

<div class="glow w-90 h-90 bg-orange-500 bottom-0 -right-20"></div>

<div class="kicker">Best use of Gemma</div>

# Gemma, everywhere

<div class="mt-8 text-xl leading-relaxed">

**Gemma 4 E2B** answers locally for free —<br>
chosen over a bigger sibling *on measured evidence*.

<div class="mt-4"></div>

**Fireworks Gemma 4** (26B-A4B / 31B) handles the escalations —<br>
with thinking off and prompts tuned per category.

</div>

<div v-click class="mt-10 opacity-60">
one model family · two tiers · every routing decision earned by data
</div>

---
layout: center
class: text-center centered-h
---

<div class="glow w-110 h-110 bg-orange-500 -top-30 left-1/3"></div>

<h1 class="no-bar" style="font-size:4rem;"><span class="num-hot">Token</span><span class="num-warm">Router</span></h1>

<div class="text-2xl mt-2 opacity-90">
Prove it for free — or buy the minimum.
</div>

<div class="mt-12 text-sm opacity-70 leading-loose">

**Go** orchestrator · **llama.cpp** + Gemma 4 E2B in-container · **Fireworks AI** escalation<br>
3.1 GB image · 50 tests incl. fuzz · every claim in this deck is logged in **eval/PERF.md**

</div>

<div class="mt-8 text-sm opacity-50 font-mono">
github.com/omerdduran/token-router
</div>
