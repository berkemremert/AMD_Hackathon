"""Pure-Python task classifier. Zero API calls — regex/keyword heuristics only.

Buckets: factual, math, sentiment, summarization, ner, code_debug, logic,
code_gen, general (fallback).
"""

import re

# Fenced code blocks or obvious code syntax in the prompt body.
_CODE_BLOCK_RE = re.compile(r"```|\bdef \w+\(|\bfunction \w+\(|#include\s*<|\bpublic (?:static )?\w+ \w+\(")

_MATH_EXPR_RE = re.compile(r"\d+\s*[\+\-\*/\^%]\s*\d+|\b\d+\s*=\s*|\\frac|\\sqrt|\bsolve for\b")

# Each entry: (category, compiled regex). Checked in order; first match wins,
# so more specific categories come before broader ones.
_RULES = [
    ("summarization", re.compile(
        r"\bsummar(?:y|ize|ise|ies)\b|\btl;?dr\b|\bcondense\b|\bshorten (?:this|the)\b|\bmain points? of\b|\bkey takeaways?\b",
        re.I)),
    ("sentiment", re.compile(
        r"\bsentiment\b|\bpositive(?:,| or |/)\s*(?:negative|neutral)\b|\bnegative or positive\b|\btone of (?:this|the)\b|\bemotion(?:al tone)?\b.*\b(?:review|text|tweet|sentence)\b",
        re.I)),
    ("ner", re.compile(
        r"\bnamed entit(?:y|ies)\b|\bextract (?:all )?(?:the )?(?:entities|names?|people|persons|organi[sz]ations?|locations?|dates?|places)\b|\bidentify (?:all )?(?:the )?(?:entities|people|persons|organi[sz]ations?|locations?)\b|\bNER\b",
        re.I)),
    ("code_debug", re.compile(
        r"\b(?:fix|debug)\b.*\b(?:code|bug|error|function|script|program)\b|\b(?:bug|error|exception|traceback|stack trace)\b.*\bcode\b|\bwhy (?:does|is|isn't|doesn't)\b.*\bcode\b|\bcode\b.*\bnot work|\bsyntax error\b|\bfind the bug\b|\bwhat'?s wrong with\b",
        re.I | re.S)),
    ("code_gen", re.compile(
        r"\bwrite (?:a |an |some |the )?(?:python|java(?:script)?|c\+\+|c#|go|rust|sql|bash|shell|typescript|ruby)?\s*(?:code|function|program|script|class|method|query|regex|snippet)\b|\bimplement (?:a|an|the)\b|\bcreate a (?:function|program|script|class)\b|\bcode (?:that|to|which)\b",
        re.I)),
    ("math", re.compile(
        r"\b(?:calculate|compute|evaluate|simplify)\b|\bhow (?:many|much)\b|\bwhat is \d|\bsolve\b.*\b(?:equation|for [a-z]|problem)\b|\bderivative\b|\bintegral\b|\bprobability\b|\bpercent(?:age)?\b|\b(?:sum|product|difference|quotient|remainder|average|mean|median) of\b|\barea of\b|\bperimeter\b|\bvolume of\b",
        re.I)),
    ("logic", re.compile(
        r"\briddle\b|\blogic (?:puzzle|problem)\b|\bpuzzle\b|\btruth[- ]?teller\b|\bknights? and knaves?\b|\bif all\b.*\bthen\b|\bwho is lying\b|\bdeduce\b|\bsyllogism\b|\bvalid (?:argument|conclusion)\b|\bfollows? logically\b",
        re.I | re.S)),
    ("factual", re.compile(
        r"^(?:who|what|when|where|which|whom|whose|why|how)\b|\bcapital of\b|\bin what year\b|\bwho (?:was|is|were|wrote|invented|discovered|founded)\b|\bname the\b|\btrue or false\b",
        re.I)),
]


def classify(prompt: str) -> str:
    """Return the category bucket for a task prompt."""
    if not prompt or not prompt.strip():
        return "general"
    text = prompt.strip()

    has_code = bool(_CODE_BLOCK_RE.search(text))

    # Code containing prompts: decide debug vs gen before generic rules,
    # since code snippets often contain question words that trip "factual".
    if has_code:
        if re.search(r"\b(?:fix|debug|bug|error|wrong|broken|not work|fail|incorrect|issue)\b", text, re.I):
            return "code_debug"
        if re.search(r"\b(?:write|create|generate|build|complete|finish|implement|extend|modify|refactor|optimi[sz]e|translate|convert|rewrite)\b", text, re.I):
            return "code_gen"
        return "code_debug"

    for category, pattern in _RULES:
        if pattern.search(text):
            return category

    # Bare arithmetic like "17 * 23 = ?" with no keyword.
    if _MATH_EXPR_RE.search(text):
        return "math"

    return "general"
