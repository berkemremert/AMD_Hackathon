"""
Per-category prompt tightening — the main token-saving lever.

Total tokens = prompt tokens + completion tokens. We cut both:
  1. classify_category(prompt): figures out WHICH of the 8 task types this is,
     using plain Python pattern-matching — costs ZERO tokens.
  2. build_remote_request(task): returns the tightest instruction + the smallest
     output cap that still protects accuracy for that category.

Category cheat-sheet (from the participant guide):
  factual    - explain concepts/definitions          -> short direct answer
  math       - multi-step arithmetic, word problems  -> brief working, then answer
  sentiment  - label + justify                       -> label + ONE sentence
  summary    - condense to a format/length           -> follow the stated constraint only
  ner        - extract entities + labels             -> compact JSON only
  debug      - find bug + corrected code             -> one-line bug + fixed code
  logic      - constraint puzzles                    -> brief reasoning, then answer
  codegen    - write a function from a spec          -> code only

Why not one-word answers everywhere? The accuracy gate is an LLM-judge checking
"expected intent". Sentiment explicitly requires justification; math/logic need
enough working to land on the right answer. So each cap is the smallest that
does NOT endanger the accuracy gate.
"""

import re

# ---------------------------------------------------------------------------
# 1. Zero-token category detection
# ---------------------------------------------------------------------------

# Checked in order; first match wins. Order matters: the most distinctive
# patterns go first so generic words like "explain" don't steal them.
_RULES = [
    ("ner", [
        r"\bnamed entit", r"\bentit(y|ies)\b.*\b(extract|label|identif)",
        r"\b(extract|label|identify)\b.*\bentit(y|ies)\b",
        r"\b(person|organization|organisation|location|date)s?\b.*\bextract",
        r"\bNER\b",
    ]),
    ("sentiment", [
        r"\bsentiment\b", r"\bclassify\b.*\b(positive|negative|neutral)",
        r"\b(positive|negative|neutral)\b.*\bclassify",
    ]),
    ("summary", [
        r"\bsummar(y|ise|ize|isation|ization)\b", r"\bcondense\b",
        r"\bin one sentence\b", r"\btl;?dr\b",
    ]),
    ("debug", [
        r"\b(bug|bugs|buggy)\b", r"\bdebug", r"\bfix (the|this) (code|function|snippet)",
        r"\bwhat('s| is) wrong with (the|this) (code|function)", r"\berror in (the|this) code\b",
    ]),
    ("codegen", [
        r"\bwrite (a|the) (python |javascript |java |c\+\+ )?(function|method|class|script|program)\b",
        r"\bimplement (a|the)\b", r"\bcode (a|the) function\b",
        r"\bdef [a-z_]+\(", r"\bfunction that\b",
    ]),
    ("math", [
        r"\bcalculate\b", r"\bhow (much|many|long|far)\b.*\d", r"\d.*\bhow (much|many|long|far)\b",
        r"\d+\s*%", r"\bpercent",
        r"\bsum of\b", r"\bproduct of\b", r"\bproject(ed|ion)\b.*\d",
        r"\d+\s*[\+\-\*/x]\s*\d+", r"\bwhat is \d", r"\baverage\b.*\d", r"\btotal\b.*\d",
    ]),
    ("logic", [
        r"\bpuzzle\b", r"\briddle\b", r"\bif .* then .* (who|what|which)\b",
        r"\b(everyone|nobody|exactly one|at least one)\b.*\b(lies|truth|liar)",
        r"\bconstraints?\b.*\bsatisf", r"\bseated? (next to|beside|left of|right of)\b",
        r"\bolder than\b.*\byounger than\b", r"\bknights? and knaves?\b",
        r"\beach (own|owns|has|have|is|are)\b.*\b(who|what|which)\b",
        r"\bwho (owns|has|is|sits|finished|won)\b",
        r"\b(does not|doesn't|not) (own|have|sit|like)\b.*\b(who|what|which)\b",
    ]),
    ("factual", [
        r"\bwhat (is|are|was|were)\b", r"\bexplain\b", r"\bdefine\b", r"\bdefinition\b",
        r"\bhow (does|do|did) .* work\b", r"\bwho (is|was)\b", r"\bwhy (is|does|do)\b",
        r"\bdescribe\b",
    ]),
]

_COMPILED = [(cat, [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats])
             for cat, pats in _RULES]


def classify_category(prompt: str) -> str:
    """Return one of the 8 category names, or 'general' if nothing matches."""
    for cat, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(prompt):
                return cat
    return "general"


# ---------------------------------------------------------------------------
# 2. Per-category tight instructions + output caps
# ---------------------------------------------------------------------------
# Instructions are deliberately SHORT (they cost prompt tokens on every call).
# max_tokens is the smallest cap that doesn't endanger the accuracy gate.

_CATEGORY_CONFIG = {
    "factual": {
        "instruction": "Answer directly and concisely. No preamble.",
        "max_tokens": 160,
    },
    "math": {
        "instruction": "Show minimal working, then end with: Answer: <result>.",
        "max_tokens": 600,
    },
    "sentiment": {
        "instruction": "State the sentiment label (positive, negative, neutral, or mixed), then one brief sentence of justification. Keep it under 30 words.",
        "max_tokens": 120,
    },
    "summary": {
        "instruction": "Follow the stated length/format exactly. Output only the summary.",
        "max_tokens": 200,
    },
    "ner": {
        "instruction": 'Output only JSON: [{"entity":"...","type":"..."}]. No other text.',
        "max_tokens": 200,
    },
    "debug": {
        "instruction": "State the bug in one line, then give the corrected code. Nothing else.",
        "max_tokens": 800,
    },
    "logic": {
        "instruction": "Solve step by step, then end with a final line: Answer: <result>.",
        "max_tokens": 800,
    },
    "codegen": {
        "instruction": "Write the complete Python function inside a single code block. Keep it concise.",
        "max_tokens": 900,
    },
    "general": {
        "instruction": "Answer correctly and concisely. No preamble.",
        "max_tokens": 400,
    },
}


def build_remote_request(task):
    """Return (messages, max_tokens, category) for the tightest safe Fireworks call."""
    prompt = task["prompt"]
    category = classify_category(prompt)
    cfg = _CATEGORY_CONFIG[category]
    # One user message: the tight instruction + the task. (A separate system
    # message adds structural token overhead on some models for no gain here.)
    messages = [{"role": "user", "content": f"{cfg['instruction']}\n\n{prompt}"}]
    return messages, cfg["max_tokens"], category
