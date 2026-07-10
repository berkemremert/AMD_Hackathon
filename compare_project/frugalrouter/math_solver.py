from __future__ import annotations

import ast
import operator
import re


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def solve_simple_math(prompt: str) -> tuple[str, str] | None:
    text = prompt.strip()
    lower = text.lower()

    percent = _solve_percentage(lower)
    if percent is not None:
        return percent, "percentage_pattern"

    if not _looks_like_direct_math_request(lower):
        return None

    expression = _extract_expression(text)
    if not expression:
        return None

    if not _is_pure_math_prompt(lower, expression):
        return None

    try:
        value = _safe_eval(expression)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None

    return _format_number(value), "direct_arithmetic_expression"


FILLER_WORDS = {
    "calculate", "compute", "evaluate", "solve", "what", "is", "the", "value",
    "of", "result", "answer", "return", "give", "only", "number", "please",
    "equals", "equal", "to", "final", "exact", "just", "then", "and", "a", "as",
    "expression", "following", "this",
}


def _is_pure_math_prompt(lower_text: str, expression: str) -> bool:
    """Fire only when the prompt is nothing but a computation request.

    Prompts like "What is 12/25 known as?" contain an evaluable span but are
    not arithmetic questions; any residual word outside the known filler
    vocabulary means the expression's meaning is uncertain, so stay remote.
    """
    remainder = lower_text.replace(expression.lower(), " ")
    words = re.findall(r"[a-z']+", remainder)
    return all(word in FILLER_WORDS for word in words)


def _looks_like_direct_math_request(text: str) -> bool:
    markers = [
        "calculate",
        "compute",
        "evaluate",
        "solve",
        "what is",
        "return only the number",
        "give only the number",
    ]
    return any(marker in text for marker in markers)


def _extract_expression(prompt: str) -> str | None:
    compact = prompt.replace("×", "*").replace("÷", "/")
    candidates = re.findall(r"[-+*/().\d\s]{3,}", compact)
    candidates = [candidate.strip() for candidate in candidates if re.search(r"\d", candidate)]
    candidates = [candidate for candidate in candidates if re.search(r"[-+*/]", candidate)]
    if not candidates:
        return None
    return max(candidates, key=len)


def _solve_percentage(text: str) -> str | None:
    # Chained forms like "20% of 50% of 200" only match the first clause and
    # would compute the wrong value; leave them to the remote model.
    if len(re.findall(r"(?:%|percent)\s+of\b", text)) > 1:
        return None

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if match and _percentage_prompt_is_pure(text, match):
        return _format_number(float(match.group(1)) * float(match.group(2)) / 100)

    match = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+what\s+(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)", text)
    if match and float(match.group(2)) != 0 and _percentage_prompt_is_pure(text, match):
        return _format_number(float(match.group(1)) * 100 / float(match.group(2)))

    match = re.search(
        r"(?:increase|raise)\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
    )
    if match and _percentage_prompt_is_pure(text, match):
        base = float(match.group(1))
        pct = float(match.group(2))
        return _format_number(base * (1 + pct / 100))

    match = re.search(
        r"(?:decrease|reduce|discount)\s+(-?\d+(?:\.\d+)?)\s+by\s+(-?\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
    )
    if match and _percentage_prompt_is_pure(text, match):
        base = float(match.group(1))
        pct = float(match.group(2))
        return _format_number(base * (1 - pct / 100))

    return None


def _percentage_prompt_is_pure(text: str, match: re.Match[str]) -> bool:
    """Reject percentage clauses embedded in a larger word problem."""
    remainder = f"{text[: match.start()]} {text[match.end():]}"
    words = re.findall(r"[a-z']+", remainder)
    return all(word in FILLER_WORDS or word == "percent" for word in words)


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"

