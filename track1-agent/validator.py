"""Pure-Python answer validation. Zero LLM calls.

validate() returns (ok, reason). A False verdict triggers main.py's single
retry (thinking ON, generous max_tokens). Checks are deliberately
lenient where counting is fuzzy — a false failure only costs one retry, but
a false pass costs judge accuracy.
"""

import ast
import json
import re

_FENCE_RE = re.compile(r"```([\w+#-]*)\n?(.*?)```", re.S)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_NUM_RE = re.compile(r"-?[\d,]*\d(?:\.\d+)?")
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
              "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _to_n(token):
    token = token.lower()
    return int(token) if token.isdigit() else _NUM_WORDS.get(token)


def _code_blocks(text):
    return [(lang.lower(), code.strip()) for lang, code in _FENCE_RE.findall(text)]


def _balanced(code):
    return all(code.count(a) == code.count(b) for a, b in ("()", "[]", "{}"))


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _check_code(prompt, answer):
    blocks = _code_blocks(answer)
    prompt_l = prompt.lower()
    python_task = "python" in prompt_l or bool(re.search(r"```python|\bdef \w+\(", prompt))

    if blocks:
        # validate the last block of the dominant language (debug answers often
        # show buggy-then-fixed code; the fix comes last)
        lang, code = blocks[-1]
        if not code:
            return False, "empty code block"
        if lang == "python" or (not lang and python_task):
            try:
                ast.parse(code)
            except SyntaxError as e:
                return False, f"python syntax error: {e.msg}"
            return True, ""
        return (True, "") if _balanced(code) else (False, "unbalanced brackets")

    # no fences: pure code answers start like code; prose answers get a
    # lenient balance check only
    head = answer.lstrip()
    if python_task and re.match(r"(?:def |class |import |from |@)", head):
        try:
            ast.parse(answer)
        except SyntaxError as e:
            return False, f"python syntax error: {e.msg}"
        return True, ""
    return (True, "") if _balanced(answer) else (False, "unbalanced brackets")


def _check_ner(answer):
    candidates = [answer.strip()]
    candidates += [code for _, code in _code_blocks(answer)]
    start, end = answer.find("["), answer.rfind("]")
    if 0 <= start < end:
        candidates.append(answer[start:end + 1])
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, (list, dict)):
            return True, ""
    return False, "no valid JSON found"


def _check_math(answer):
    for line in reversed(answer.strip().splitlines()):
        if line.strip():
            return (True, "") if _NUM_RE.search(line) else (False, "no number in final line")
    return False, "no final line"


def _check_summarization(prompt, answer):
    prompt_l = prompt.lower()
    text = re.sub(r"^\s*tl;?dr:?\s*", "", answer.strip(), flags=re.I)

    m = re.search(r"exactly (\w+) bullet points?", prompt_l)
    if m:
        want = _to_n(m.group(1))
        got = len(_BULLET_RE.findall(text))
        if want and got != want:
            return False, f"expected {want} bullets, got {got}"
        return True, ""

    m = re.search(r"(?:no more than|at most|maximum of) (\d+) words", prompt_l)
    if m:
        limit = int(m.group(1))
        got = len(text.split())
        if got > limit:
            return False, f"{got} words > limit {limit}"
        return True, ""

    m = re.search(r"exactly (\w+) sentences?", prompt_l) or re.search(r"in exactly (\w+) sentences?", prompt_l)
    if m:
        want = _to_n(m.group(1))
        got = len(_sentences(text))
        if want and got != want:
            return False, f"expected {want} sentences, got {got}"
    return True, ""


def validate(category, prompt, answer, finish_reason=None):
    """Return (ok, reason). Applies to every category: empty/None and
    truncated (finish_reason=length) answers always fail."""
    if answer is None or not str(answer).strip():
        return False, "empty answer"
    answer = str(answer)
    if finish_reason == "length":
        return False, "truncated (finish_reason=length)"

    if category == "entity_extraction":
        return _check_ner(answer)
    if category in ("code_authoring", "bug_fixing"):
        return _check_code(prompt, answer)
    if category == "math_solving":
        return _check_math(answer)
    if category == "summarization":
        return _check_summarization(prompt, answer)
    return True, ""
