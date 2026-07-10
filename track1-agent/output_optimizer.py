"""
Dynamic Output Optimizer
This module detects the task category using fast heuristics and applies an optimal
system instruction and output token cap to minimize API costs without losing accuracy.
"""

import re

# We define regex heuristics to quickly classify tasks into 8 domains.
# The order here is optimized to catch the most specific domains first.
HEURISTICS = {
    "entity_extraction": [
        r"\bnamed entit", r"\bentit(y|ies)\b.*\b(extract|label|identif)",
        r"\b(extract|label|identify)\b.*\bentit(y|ies)\b",
        r"\b(person|organization|organisation|location|date)s?\b.*\bextract",
        r"\bNER\b",
    ],
    "sentiment_analysis": [
        r"\bsentiment\b", r"\bclassify\b.*\b(positive|negative|neutral)",
        r"\b(positive|negative|neutral)\b.*\bclassify",
    ],
    "summarization": [
        r"\bsummar(y|ise|ize|isation|ization)\b", r"\bcondense\b",
        r"\bin one sentence\b", r"\btl;?dr\b",
    ],
    "bug_fixing": [
        r"\b(bug|bugs|buggy)\b", r"\bdebug", r"\bfix (the|this) (code|function|snippet)",
        r"\bwhat('s| is) wrong with (the|this) (code|function)", r"\berror in (the|this) code\b",
    ],
    "code_authoring": [
        r"\bwrite (a|the) (python |javascript |java |c\+\+ )?(function|method|class|script|program)\b",
        r"\bimplement (a|the)\b", r"\bcode (a|the) function\b",
        r"\bdef [a-z_]+\(", r"\bfunction that\b",
    ],
    "math_solving": [
        r"\bcalculate\b", r"\bhow (much|many|long|far)\b.*\d", r"\d.*\bhow (much|many|long|far)\b",
        r"\d+\s*%", r"\bpercent",
        r"\bsum of\b", r"\bproduct of\b", r"\bproject(ed|ion)\b.*\d",
        r"\d+\s*(?:[\+\*/x]|\s-\s)\s*\d+", r"\bwhat is \d", r"\baverage\b.*\d", r"\btotal\b.*\d",
    ],
    "logical_puzzles": [
        r"\bpuzzle\b", r"\briddle\b", r"\bif .* then .* (who|what|which)\b",
        r"\b(everyone|nobody|exactly one|at least one)\b.*\b(lies|truth|liar)",
        r"\bconstraints?\b.*\bsatisf", r"\bseated? (next to|beside|left of|right of)\b",
        r"\bolder than\b.*\byounger than\b", r"\bknights? and knaves?\b",
        r"\beach (own|owns|has|have|is|are)\b.*\b(who|what|which)\b",
        r"\bwho (owns|has|is|sits|finished|won)\b",
        r"\b(does not|doesn't|not) (own|have|sit|like)\b.*\b(who|what|which)\b",
    ],
    "knowledge_qa": [
        r"\bwhat (is|are|was|were)\b", r"\bexplain\b", r"\bdefine\b", r"\bdefinition\b",
        r"\bhow (does|do|did) .* work\b", r"\bwho (is|was)\b", r"\bwhy (is|does|do)\b",
        r"\bdescribe\b",
    ],
}

COMPILED_HEURISTICS = {
    domain: [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in patterns]
    for domain, patterns in HEURISTICS.items()
}

def detect_task_type(user_prompt: str) -> str:
    """Scans the prompt and returns the detected task domain. Returns 'fallback' if unknown."""
    for domain, compiled_patterns in COMPILED_HEURISTICS.items():
        for pattern in compiled_patterns:
            if pattern.search(user_prompt):
                return domain
    return "fallback"

# Optimized constraints for each domain to ensure maximum token savings 
# without triggering a failure from the LLM-as-a-judge accuracy gate.
_BASE = "Answer in English. Be concise and direct; no preamble, no restating the question."

TOKEN_LIMITS = {
    "knowledge_qa": {
        "system": f"{_BASE} Give a correct, clear answer in under 120 words.",
        "cap": 300,
        "retry_cap": 1024,
    },
    "math_solving": {
        "system": f"{_BASE} Work through it in brief steps, then end with 'Answer: <value>' on its own line.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "sentiment_analysis": {
        "system": f"{_BASE} State the sentiment as positive, negative, or neutral, then one short reason.",
        "cap": 150,
        "retry_cap": 300,
    },
    "summarization": {
        "system": f"{_BASE} Output only the summary and obey any length or format constraint stated in the task.",
        "cap": 300,
        "retry_cap": 600,
    },
    "entity_extraction": {
        "system": f"{_BASE} Output valid JSON only: [{{\"entity\":\"...\",\"type\":\"...\"}}]. Do not output conversational text.",
        "cap": 300,
        "retry_cap": 600,
    },
    "bug_fixing": {
        "system": f"{_BASE} State the bug in one sentence, then give the corrected code in a single fenced block.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "logical_puzzles": {
        "system": f"{_BASE} Reason in brief numbered steps, checking each constraint, then end with 'Answer: <value>' on its own line.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "code_authoring": {
        "system": f"{_BASE} Output only the code in a single fenced block — correct, complete, and self-contained.",
        "cap": 768,
        "retry_cap": 1536,
    },
    "fallback": {
        "system": f"{_BASE}",
        "cap": 400,
        "retry_cap": 1024,
    },
}
