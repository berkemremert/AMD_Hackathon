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
TOKEN_LIMITS = {
    "knowledge_qa": {
        "suffix": "Provide a direct, concise answer without any preamble.",
        "cap": 300,
        "retry_cap": 1024,
    },
    "math_solving": {
        "suffix": "Briefly show your steps, then conclude with: Answer: <result>.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "sentiment_analysis": {
        "suffix": "Provide the exact sentiment label (positive, negative, neutral, mixed) followed by a single sentence of justification.",
        "cap": 150,
        "retry_cap": 300,
    },
    "summarization": {
        "suffix": "Provide only the summary, adhering strictly to the requested format or length.",
        "cap": 300,
        "retry_cap": 600,
    },
    "entity_extraction": {
        "suffix": 'Output valid JSON only: [{"entity":"...","type":"..."}]. Do not output conversational text.',
        "cap": 300,
        "retry_cap": 600,
    },
    "bug_fixing": {
        "suffix": "Identify the bug in one sentence, then provide the corrected code block. No extra text.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "logical_puzzles": {
        "suffix": "Provide brief step-by-step reasoning, then conclude with: Answer: <result>.",
        "cap": 512,
        "retry_cap": 1024,
    },
    "code_authoring": {
        "suffix": "Write the requested code inside a single block. Be as concise as possible.",
        "cap": 768,
        "retry_cap": 1536,
    },
    "fallback": {
        "suffix": "Answer the question directly and concisely.",
        "cap": 400,
        "retry_cap": 1024,
    },
}
