# Phase 1 — Category 1 (Factual Knowledge) Pilot Dataset
**Target: 50 prompt+answer+difficulty entries, built entirely from local TriviaQA data. No Fireworks or any LLM calls anywhere in this pipeline.**

This is a pilot batch to validate the format, sourcing approach, and difficulty heuristic before we scale to the full Category 1 dataset. Treat every threshold below as a first guess — flag anything that looks wrong rather than silently working around it.

---

## Ground rules

- **No API calls.** No Fireworks, no OpenAI, no local LLM inference either. This batch is pure data engineering against the TriviaQA file already on disk.
- **English only.**
- **No duplicate prompts** (exact or near-duplicate).
- **Every entry must be traceable** back to its original TriviaQA question ID.
- **Document every heuristic decision** (especially difficulty labeling) in a README — we will revisit and likely rewrite these thresholds after reviewing the pilot.
- Do not touch Categories 2–8 or scale past 50 in this phase.

---

## Step 0 — Inventory the source before writing any pipeline code

TriviaQA ships in multiple variants (`rc` vs `unfiltered`, `wikipedia` vs `web` evidence, train/dev/test splits) with slightly different schemas. Before building anything:

1. Identify exactly which file(s) you have (check filename, and the top-level structure — is it the `rc` or `unfiltered` set? Does it include an `EntityPages`/evidence field?).
2. Print/log the full field list from 5 sample records to a `notes/schema_inventory.md` file.
3. Report back to me which variant it is — this affects how reliable the "answer" field is and matters later for eval integrity.

---

## Step 1 — Build the candidate pool

Filter TriviaQA questions down to a clean candidate set:

- Keep only entries with a **single, unambiguous primary answer** (skip anything with wildly divergent aliases suggesting an ambiguous/compound answer).
- Keep only entries with **Wikipedia-sourced evidence** if the field is available (more reliable ground truth than open-web evidence).
- Question length: reasonable single-sentence trivia questions (rough guideline: 5–40 words) — drop extreme outliers.
- Drop anything that reads as offensive, violent, or otherwise inappropriate.

## Step 2 — Deduplicate

Normalize question text (lowercase, strip punctuation/whitespace) and drop exact or near-duplicate questions (simple string similarity is fine for this scale).

## Step 3 — Stratified topic sampling (for diversity)

Tag each candidate with a coarse topic bucket using simple keyword/entity heuristics — e.g. `people`, `places`, `history`, `science`, `sports`, `entertainment`, `literature`, `other`. When selecting the final 50:

- Cover **at least 5 distinct buckets**.
- No single bucket should exceed ~15 of the 50 entries.

## Step 4 — Difficulty labeling (heuristic, no model call)

Since we're not calling an LLM to judge difficulty, use a transparent, computable heuristic. Suggested starting point (adjust if TriviaQA's fields give you something better):

- **Easy**: answer has many aliases listed (proxy for a well-known/famous entity) and/or short, common answer string.
- **Hard**: answer has few/no aliases, longer or more obscure answer string.
- **Medium**: everything in between.

Aim for a roughly even 3-way split across the 50. **Explicitly label this heuristic as provisional** in the README — we'll sanity-check it against manual judgment after the pilot.

## Step 5 — Format transform

Each final entry should look like this:

```json
{
  "task_id": "cat1_pilot_0001",
  "category": "factual_knowledge",
  "prompt": "<question text, lightly cleaned>",
  "reference_answer": "<primary answer string>",
  "answer_aliases": ["...", "..."],
  "difficulty": "easy",
  "source": "triviaqa",
  "source_id": "<original TriviaQA question id>"
}
```

Note: `reference_answer`, `answer_aliases`, `difficulty`, `source`, and `source_id` are for **your internal dev/testing use only** — they are not part of the actual hackathon harness input format (`task_id` + `prompt` only). Keep this pilot file clearly separate from anything shaped like the real `/input/tasks.json`.

## Step 6 — Manual spot-check

50 is small enough to eyeball. Before calling this done, review every entry for:
- Correctness of the answer
- No genuinely ambiguous/multi-valid-answer questions slipped through
- No inappropriate content
- Difficulty labels look sane on manual read

## Deliverables

- `/data/category1_pilot_v1.json` — the 50 entries
- `/data/category1_pilot_v1_README.md` — source file/variant used, filter thresholds, difficulty heuristic definition, known limitations, and topic bucket counts
- `/scripts/build_category1_pilot.py` — reproducible script (not one-off/manual), with sample size as a configurable constant so we can rerun it at larger scale later without a rewrite

## Report back to me on

1. Which exact TriviaQA variant/split you used.
2. TriviaQA's license terms — right now we're treating this dataset as **internal dev/test only, not part of the public hackathon repo**. Confirm that's still the plan, since the license matters a lot more if it ever gets committed publicly.
3. A known coverage gap: TriviaQA skews toward discrete "who/what/when/where" recall questions. Category 1 also includes "explaining how things work"-style prompts, which TriviaQA doesn't really cover. We'll need a plan for that gap before phase 2 — flagging now so it's not a surprise later.