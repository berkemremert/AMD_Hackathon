"""Zero-token task classification.

A single regex pass over the prompt assigns one of eight task categories.
No model call is involved, so routing itself costs nothing against the
scored token budget. Categories are checked most-specific first; anything
that matches nothing falls back to FACTUAL (the most general handler).
"""

from __future__ import annotations

import re
from enum import Enum


class Category(str, Enum):
    FACTUAL = "factual"
    MATH = "math"
    SENTIMENT = "sentiment"
    SUMMARIZATION = "summarization"
    NER = "ner"
    CODE_DEBUG = "code_debug"
    CODE_GEN = "code_gen"
    LOGIC = "logic"


_CODE_FENCE = re.compile(r"```")
_CODE_TOKENS = re.compile(
    r"\b(def |class |return |import |#include|public |void |printf|"
    r"console\.log|System\.out)|=>|;\s*$",
    re.MULTILINE,
)

# Ordered patterns, checked top to bottom. First category with any hit wins,
# so a review that also says "what is" still lands on SENTIMENT.
_PATTERNS: list[tuple[Category, list[str]]] = [
    (Category.CODE_DEBUG, [
        r"\bbug\b", r"\bdebug\b", r"\bfix (this|the|my|it)\b",
        r"what'?s wrong", r"why (does|is)n'?t (this|it|my)\b",
        r"error in (this|the|my)\b", r"traceback", r"stack ?trace",
        r"throws? an? (error|exception)", r"returns? \w+ instead",
        r"infinite loop", r"corrected (version|code)",
    ]),
    (Category.CODE_GEN, [
        r"\b(write|create|implement|build|generate|produce|give me)\b.*"
        r"\b(function|method|class|program|script|routine)\b",
        r"\bfunction (that|to)\b", r"\bcode that\b",
        r"\bwrite (a|an|some) code\b", r"\bimplement (a|an|the)\b",
    ]),
    (Category.SENTIMENT, [
        r"\bsentiment\b", r"positive or negative", r"positive, negative",
        r"classify the (tone|emotion|sentiment|mood)",
        r"(emotional )?tone of (this|the|that)",
        r"\b(positive|negative|neutral)\b.*\breview\b",
        r"how (positive|negative) ", r"is this (review|tweet|comment)\b",
    ]),
    (Category.NER, [
        r"named entit", r"\bner\b",
        r"extract (all )?(the )?(entit|name|person|people|organi|location|date)",
        r"(list|identify|find|pull out) (all )?(the )?"
        r"(people|persons?|organi[sz]ations?|locations?|dates?|entit)",
        r"(person|organization|location|date)\s*[:=]",
    ]),
    (Category.SUMMARIZATION, [
        r"summari[sz]e", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
        r"\bshorten\b", r"in (one|a single|two|three|\d+) (sentences?|words?|lines?)",
        r"main (idea|point|takeaway)", r"\bthe gist\b", r"key points",
        r"boil .* down",
    ]),
    (Category.LOGIC, [
        r"\bpuzzle\b", r"who (is|owns|sits|lives|has|drinks|likes)\b",
        r"if and only if", r"exactly one", r"at least one",
        r"the following (clues|facts|statements|conditions)",
        r"each (person|house|box|day|one) .*(different|exactly|only|one)",
        r"\bdeduce\b", r"logically (follows?|true)",
        r"(definitely|necessarily) (true|follows)",
        r"knights? and knaves", r"truth[- ]?teller", r"\bliar\b",
    ]),
    (Category.MATH, [
        r"\bcalculate\b", r"\bcompute\b", r"how (much|many)\b", r"percent",
        r"\d+\s*%", r"\bsum of\b", r"\baverage\b", r"solve for\b",
        r"\d+\s*[+\-*/x×÷]\s*\d+", r"total (cost|price|amount)",
        r"\b(interest|discount|ratio|profit)\b",
        r"find the (largest|smallest|value|angle|area|sum|total|average)",
        r"what is \d",
    ]),
    (Category.FACTUAL, [
        r"what (is|are|was|were)\b", r"who (is|was|were)\b",
        r"when (did|was|is)\b", r"where (is|was|are)\b",
        r"why (is|do|does|are)\b", r"how (do|does|can)\b",
        r"\bexplain\b", r"\bdefine\b", r"\bdescribe\b", r"what does .* mean",
    ]),
]

_COMPILED = [
    (cat, [re.compile(p, re.IGNORECASE) for p in pats])
    for cat, pats in _PATTERNS
]


def _has_code(text: str) -> bool:
    return bool(_CODE_FENCE.search(text) or _CODE_TOKENS.search(text))


def classify(prompt: str) -> Category:
    text = prompt or ""
    for cat, rxs in _COMPILED:
        if any(rx.search(text) for rx in rxs):
            return cat
    # No keyword hit: a bare code block is almost always a debugging task.
    return Category.CODE_DEBUG if _has_code(text) else Category.FACTUAL
