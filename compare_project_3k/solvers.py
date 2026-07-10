"""Zero-token deterministic solvers.

Some tasks can be answered by plain code with certainty — a logic puzzle
brute-forced over every assignment, an arithmetic expression evaluated
exactly. When one of these fires it costs zero scored tokens and the answer
cannot be wrong, so it both saves tokens and locks that task's correctness
against judge noise.

Every solver is strictly self-gating: on any ambiguity, unexpected phrasing,
or non-unique result it returns None and the task falls through to the model.
A deferred task just pays normal tokens; a wrongly-fired solver would cost an
answer, so the bias is heavily toward deferring.
"""

from __future__ import annotations

import re
from itertools import permutations

# --- Single-domain assignment logic puzzles ---------------------------------
# Shape: N named people each own a distinct value from ONE domain, plus
# owns / does-not-own clues, then "Who owns the X?" or "What does X own?".
# Restricted to ownership verbs (own/have/possess) so the restated answer is
# always phrased correctly; other relations (lives in, drinks...) defer.

_REL = re.compile(r"\b(owns?|have|has|possess(?:es)?)\b", re.IGNORECASE)
_NEG = re.compile(r"\b(not|never)\b|n['’]t", re.IGNORECASE)

# Words that can look like a name (capitalized) but are not people.
_NOT_NAMES = {
    "the", "a", "an", "who", "what", "which", "when", "where", "why", "how",
    "if", "each", "every", "no", "none", "one", "two", "three", "four", "five",
    "both", "neither", "either", "he", "she", "it", "they", "we", "you", "i",
    "and", "or", "but", "so", "then", "also", "here", "there", "this", "that",
    "these", "those", "their", "his", "her", "its", "there", "three", "friends",
}
_ARTICLES = {"a", "an", "the", "and", "or"}

# Domain declaration: a colon list ("... a different pet: cat, dog, and bird")
# or a single parenthetical ("... a pet (cat, dog, bird)").
_COLON_LIST = re.compile(
    r":\s*([a-z]+(?:[ ,]+[a-z]+)+)\s*(?:[.?!]|$)", re.IGNORECASE)
_PAREN_LIST = re.compile(r"\(\s*([a-z]+(?:\s*,\s*[a-z]+)+[^)]*)\)", re.IGNORECASE)


def _domain(prompt: str) -> list[str] | None:
    """The one value domain, or None if not exactly one is declared."""
    colon = _COLON_LIST.search(prompt)
    parens = _PAREN_LIST.findall(prompt)
    if colon and not parens:
        inner = colon.group(1)
    elif not colon and len(parens) == 1:
        inner = parens[0]
    else:
        return None
    values = [w.lower() for w in re.findall(r"[a-zA-Z]+", inner)
              if w.lower() not in _ARTICLES]
    return values or None


def _people(prompt: str, valset: set[str]) -> list[str]:
    people, seen = [], set()
    for w in re.findall(r"\b[A-Z][a-z]+\b", prompt):
        low = w.lower()
        if low in _NOT_NAMES or low in valset or low in seen:
            continue
        seen.add(low)
        people.append(w)
    return people


def solve_logic(prompt: str) -> str | None:
    values = _domain(prompt)
    if not values or not (2 <= len(values) <= 6):
        return None
    if len(set(values)) != len(values):
        return None
    valset = set(values)

    people = _people(prompt, valset)
    if len(people) != len(values):
        return None
    by_lower = {p.lower(): p for p in people}

    cons: list[tuple[str, str, bool]] = []      # (person, value, negated)
    queries: list[tuple[str, str]] = []          # ('who', value) | ('what', person)

    for raw in re.split(r"[.?!]", prompt):
        s = raw.strip()
        if not s or not _REL.search(s):
            continue
        low = s.lower()
        # The declaration sentence carries the value list, not a clue.
        if _COLON_LIST.search(s) or _PAREN_LIST.search(s):
            continue

        vals_in = [v for v in values if re.search(rf"\b{re.escape(v)}\b", low)]
        ppl_in = [by_lower[n] for n in by_lower
                  if re.search(rf"\b{re.escape(n)}\b", low)]

        # Query forms.
        if re.match(r"who\b", low) or re.search(r"\bwho\b", low):
            if len(vals_in) == 1 and not ppl_in:
                queries.append(("who", vals_in[0]))
                continue
            return None
        if "what" in low or "which" in low:
            if len(ppl_in) == 1 and not vals_in:
                queries.append(("what", ppl_in[0]))
                continue
            return None

        # Clue: exactly one person, at least one value.
        if len(ppl_in) != 1 or not vals_in:
            return None
        person = ppl_in[0]
        if _NEG.search(low):
            for v in vals_in:
                cons.append((person, v, True))
        else:
            if len(vals_in) != 1:
                return None
            cons.append((person, vals_in[0], False))

    if not cons or not queries:
        return None

    answer = None
    consistent = False
    n = len(people)
    for perm in permutations(range(n)):
        val_of = {people[i]: values[perm[i]] for i in range(n)}
        who_has = {values[perm[i]]: people[i] for i in range(n)}
        if any((val_of[p] == v) == neg for p, v, neg in cons):
            continue
        consistent = True
        cur = []
        for kind, x in queries:
            if kind == "who":
                cur.append(f"{who_has[x]} owns the {x}")
            else:  # 'what does X own'
                cur.append(f"{x} owns the {val_of[x]}")
        if answer is None:
            answer = cur
        elif cur != answer:
            return None  # queried cell not uniquely determined
    if not consistent or not answer:
        return None
    return ", and ".join(answer) + "."


# --- Pure arithmetic expressions --------------------------------------------
# Fires only when the whole prompt reduces to a bare arithmetic expression
# ("What is 12 * (3 + 4)?"). Word problems contain other words, fail to
# tokenize, and defer — so this can never botch a computation it shouldn't own.

_ARITH_PREFIX = re.compile(
    r"^(what\s+is|what's|calculate|compute|evaluate|what\s+does|how\s+much\s+is|solve)\b[:,]?\s*",
    re.IGNORECASE)
# Exponent is deliberately excluded: it defers rather than risk a huge-int
# blow-up, and powers are rare in these arithmetic tasks anyway.
_ARITH_CHARS = re.compile(r"^[0-9+\-*/(). ]+$")


def _eval_arith(expr: str) -> float | None:
    expr = (expr.replace("×", "*").replace("÷", "/")
            .replace(",", "").replace("$", "").strip())
    if not expr or len(expr) > 80 or not _ARITH_CHARS.match(expr):
        return None
    if not re.search(r"[+\-*/]", expr):
        return None  # a bare number is not a computation worth owning
    try:
        node = compile(expr, "<arith>", "eval")
        if node.co_names:  # no identifiers allowed at all
            return None
        val = eval(node, {"__builtins__": {}}, {})  # noqa: S307 - digits/operators only
    except (ZeroDivisionError, SyntaxError, ValueError, OverflowError, TypeError):
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _format_number(v: float) -> str:
    if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def solve_arithmetic(prompt: str) -> str | None:
    s = prompt.strip().rstrip("?=.").strip()
    s = _ARITH_PREFIX.sub("", s).strip()
    val = _eval_arith(s)
    if val is None:
        return None
    return _format_number(val)
