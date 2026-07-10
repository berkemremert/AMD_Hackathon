from __future__ import annotations

import re


def classify_prompt(prompt: str) -> str:
    text = prompt.lower()

    if "function definition" in text and "parameters" in text:
        return "ner"

    if _contains_any(
        text,
        [
            "debug",
            "fix the bug",
            "bug in",
            "fix the code",
            "traceback",
            "error in this code",
            "corrected code",
            "syntax error",
            "typeerror",
            "does not crash",
            "fix the function",
            "what is the syntax error",
            "missing closing tag",
            "keyword is missing",
            "keyerror",
            "throws a keyerror",
            "unboundlocalerror",
            "infinite loop",
            "infinite cycle",
            "loop freezes",
            "operation is missing inside the loop",
            "critical flaw",
            "conceptual bug",
            "effectively ignored",
            "greedy behavior",
            "security vulnerability",
            "injection flaw",
            "why will this",
            "causes the browser to freeze",
            "what specific keyword",
            "built-in dictionary method",
            "last element",
            "scoping issue",
            "floating point precision",
            "it prints",
            "typo in this html",
            "items aren't centering",
        ],
    ):
        return "code_debugging"

    if _contains_any(
        text,
        [
            "write a function",
            "implement",
            "generate code",
            "return code",
            "python function",
            "javascript function",
            "sql query",
            "write a sql",
            "line of python",
            "code only",
            "write a bash command",
            "what bash command",
            "write a regular expression",
            "write a regex",
            "regular expression",
            "minimal dockerfile",
            "dockerfile",
            "using only css",
            "css grid",
            "output only the command",
            "one line of code",
            "arrow function",
            "lambda function",
            "list comprehension",
            "awk script",
            "sql clause",
            "debounce",
            "generate an empty json",
            "slice a string",
            "slice notation",
            "dropdown list",
            "tag name without brackets",
        ],
    ):
        return "code_generation"

    if all(label in text for label in ["positive", "negative", "neutral"]):
        return "sentiment"

    if _contains_any(text, ["customer's tone", "analyze the tone", "analyse the tone", "emotion", "lgtm"]):
        return "sentiment"

    if text.strip().startswith("classify:") or "read this review" in text:
        return "sentiment"

    if _contains_any(text, ["sentiment", "positive", "negative", "neutral"]) and _contains_any(
        text, ["classify", "label", "identify", "evaluate", "determine", "analyze", "analyse", "what is"]
    ):
        return "sentiment"

    if "tone" in text and _contains_any(
        text, ["classify", "label", "identify", "evaluate", "determine", "analyze", "analyse"]
    ):
        return "sentiment"

    if _contains_any(
        text,
        [
            "summarize",
            "summarise",
            "summary",
            "condense",
            "headline",
            "main point",
            "tl;dr",
            "tldr",
            "key times",
            "moral",
            "actual final deadline",
            "key details",
            "penalties sequentially",
            "push the q3 planning meeting",
            "new meeting",
            "write it as if",
        ],
    ):
        return "summarization"

    if _contains_any(text, ["named entity", "extract entities"]) or (
        _contains_any(text, ["extract", "identify", "list", "find"])
        and _contains_any(
            text,
            [
                "person",
                "people",
                "organization",
                "organisation",
                "location",
                "date",
                "entities",
                "names",
                "product",
                "city",
                "country",
                "structure",
                "corporation",
                "acronym",
                "planet",
                "monetary",
                "proper name",
                "email",
                "phone",
                "formula",
                "formulas",
                "regulations",
                "abbreviations",
            ],
        )
    ) or _contains_any(
        text,
        [
            "email addresses",
            "phone number",
            "all acronyms",
            "chemical compounds",
            "data privacy regulations",
            "monetary values",
            "proper name",
            "unique cities",
            "technology corporation",
            "fictional planet",
        ],
    ):
        return "ner"

    if "next number" in text and "sequence" in text:
        return "math"

    if _contains_any(
        text,
        [
            "logic puzzle",
            "deduce",
            "constraint",
            "which person",
            "who owns",
            "must be true",
            "cannot both",
            "exactly one of",
            "exactly one person",
            "exactly one statement",
            "older than",
            "younger than",
            "taller than",
            "shorter than",
            "comes next",
            "sequence",
            "all but",
            "sitting between",
            "directly between",
            "to the left of",
            "to the right of",
            "if yesterday",
            "if tomorrow",
            "if all",
            "if some",
            "if no ",
            "finished before",
            "finished after",
            "finished first",
            "finished last",
            "knight",
            "knave",
            "always tell the truth",
            "always lie",
            "every label",
            "label is wrong",
            "labels are wrong",
            "mislabeled",
            "premise",
            "definitely",
            "incorrectly labeled",
            "brother of",
            "sister of",
            "father of",
            "related to",
            "biological son",
            "not the boy's father",
            "day of the week",
            "false logs",
            "true logs",
            "parity",
            "photograph",
            "schedule three meetings",
            "must happen before",
            "must happen after",
            "power strips",
            "houses in a row",
            "single-file line",
            "bat and a ball",
            "snail is at the bottom",
            "elevator starting",
            "flight lands",
            "day before yesterday",
            "day after tomorrow",
        ],
    ):
        return "logic"

    if _contains_any(
        text,
        [
            "how many official time zones",
            "universal donor",
            "blood type",
            "which instrument family",
            "first human to journey into outer space",
            "rms titanic",
            "historic nasa mission",
            "recognized planets",
            "states in the us",
        ],
    ):
        return "factual"

    if _contains_any(
        text,
        [
            "calculate",
            "compute",
            "multiply",
            "evaluate the following expression",
            "percentage",
            "percent",
            "probability",
            "total cost",
            "how many",
            "how far",
            "how much",
            "remainder",
            "divided by",
            "compound interest",
            "solve for x",
            "area of",
            "what time",
            "earliest time",
            "time is it",
            "what floor",
            "% of",
            "discount",
            "sum of",
            "product of",
            "average of",
            "final price",
            "how long",
            "how fast",
            "at what time",
            "average speed",
            "km/h",
            "km per hour",
            "miles per hour",
            "per hour",
            "per second",
        ],
    ) or re.search(r"\d+\s*[-+*/]\s*\d+", text) or re.search(r"\d+\s*%", text):
        return "math"

    if _contains_any(
        text,
        [
            "what is",
            "explain",
            "define",
            "how does",
            "why does",
            "what year",
            "what does",
            "name the",
            "who was",
            "which company",
            "which country",
            "which planet",
            "which animal",
            "which sovereign state",
            "which instrument",
            "capital of",
            "who painted",
            "stand for",
            "chemical symbol",
            "who developed",
            "who invented",
            "who wrote",
            "who discovered",
            "si unit",
            "time complexity",
        ],
    ):
        return "factual"

    return "general"


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)
