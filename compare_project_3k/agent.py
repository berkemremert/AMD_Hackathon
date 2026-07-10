"""Per-category answer strategy: classify, then one tuned Fireworks call.

Each category carries a terse system prompt, a token cap, and a model tier.
Prompts are deliberately short — input tokens count toward the score — and
push the model to answer directly without preamble.
"""

from __future__ import annotations

from classifier import Category, classify
from llm import complete, model_for
from solvers import solve_arithmetic, solve_logic

CHEAP, STRONG, CODE = "cheap", "strong", "code"

# Proven config (v3, 16/19 on the leaderboard): the strong tier is a
# reasoning model (minimax-m3) run with reasoning_effort=none, so it can only
# reason in VISIBLE chain-of-thought. Removing the "brief steps" instruction
# (v7) collapsed accuracy to 12/19 — the CoT that costs completion tokens is
# also what carries correctness on math/logic here. Do not trim it.
_BASE = "Answer in English. Be concise and direct; no preamble, no restating the question."

_CONFIG: dict[Category, tuple[str, int, str]] = {
    Category.FACTUAL: (
        f"{_BASE} Give a correct, clear answer in under 120 words.",
        320, STRONG,
    ),
    Category.MATH: (
        f"{_BASE} Work through it in brief steps, then end with "
        f"'Answer: <value>' on its own line.",
        400, STRONG,
    ),
    Category.SENTIMENT: (
        f"{_BASE} State the sentiment as positive, negative, or neutral, "
        f"then one short reason.",
        120, CHEAP,
    ),
    Category.SUMMARIZATION: (
        f"{_BASE} Output only the summary and obey any length or format "
        f"constraint stated in the task.",
        240, CHEAP,
    ),
    Category.NER: (
        f"{_BASE} List each entity as 'label: value', one per line, using "
        f"the labels person, organization, location, date.",
        260, CHEAP,
    ),
    Category.CODE_DEBUG: (
        f"{_BASE} State the bug in one sentence, then give the corrected "
        f"code in a single fenced block.",
        520, CODE,
    ),
    Category.CODE_GEN: (
        f"{_BASE} Output only the code in a single fenced block — correct, "
        f"complete, and self-contained.",
        520, CODE,
    ),
    Category.LOGIC: (
        f"{_BASE} Reason in brief numbered steps, checking each constraint, "
        f"then end with 'Answer: <value>' on its own line.",
        460, STRONG,
    ),
}


# Zero-token deterministic solvers, tried per category before any model call.
# Each self-gates and returns None on the slightest doubt, so a miss simply
# falls through to the model — it can never turn a gettable task into a wrong
# answer. Order-independent: at most one fires per task.
_SOLVERS = {
    Category.LOGIC: solve_logic,
    Category.MATH: solve_arithmetic,
}


def solve(prompt: str) -> str:
    category = classify(prompt)
    solver = _SOLVERS.get(category)
    if solver is not None:
        try:
            answer = solver(prompt)
        except Exception:  # a solver bug must never break the task
            answer = None
        if answer:
            return answer

    system, max_tokens, tier = _CONFIG[category]
    primary = model_for(tier)
    # Blank/failed answers retry on the opposite general tier.
    fallback = model_for(STRONG if tier == CHEAP else CHEAP)
    return complete(prompt, system=system, max_tokens=max_tokens,
                    model=primary, fallback_model=fallback)
