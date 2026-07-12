"""Small exhaustive solver for one-to-one ownership/choice puzzles."""
from __future__ import annotations

import re
from itertools import permutations


IGNORED_NAMES = {
    "the", "a", "an", "who", "what", "which", "when", "where", "why", "how",
    "if", "each", "every", "no", "none", "one", "two", "three", "four", "five",
    "both", "neither", "either", "he", "she", "it", "they", "we", "you", "i",
    "and", "or", "but", "so", "then", "also", "here", "there", "this", "that",
    "these", "those", "their", "his", "her", "its", "friends",
}
RELATION_RE = re.compile(
    r"\b(owns?|have|has|possess(?:es)?|purchased?|bought|choose|chose|gets?|likes?)\b"
)


def solve_logic_puzzle(prompt: str) -> str | None:
    items = _extract_items(prompt)
    if not 2 <= len(items) <= 6 or len(set(items)) != len(items):
        return None

    names = _extract_names(prompt, set(items))
    if len(names) != len(items):
        return None

    parsed = _parse_constraints(prompt, names, items)
    if parsed is None:
        return None
    constraints, questions = parsed

    unique_answer = None
    for assignment in permutations(items):
        mapping = dict(zip(names, assignment))
        if not all((mapping[name] == item) == expected for name, item, expected in constraints):
            continue
        reverse = {item: name for name, item in mapping.items()}
        answer = [
            f"{reverse[target]} owns the {target}" if kind == "who" else f"{target} owns the {mapping[target]}"
            for kind, target in questions
        ]
        if unique_answer is not None and answer != unique_answer:
            return None
        unique_answer = answer

    return ", and ".join(unique_answer) + "." if unique_answer else None


def _extract_items(prompt: str) -> list[str]:
    patterns = (
        r":\s*([a-z]+(?:[ ,]+[a-z]+)+)\s*(?:[.?!]|$)",
        r"\(\s*([a-z]+(?:\s*,\s*[a-z]+)+[^)]*)\)",
        r"\{([a-z]+(?:\s*,\s*[a-z]+)+[^}]*)\}",
    )
    match = next((re.search(pattern, prompt, re.IGNORECASE) for pattern in patterns if re.search(pattern, prompt, re.IGNORECASE)), None)
    if match is None:
        return []
    return [
        word.lower()
        for word in re.findall(r"[a-zA-Z]+", match.group(1))
        if word.lower() not in {"a", "an", "the", "and", "or"}
    ]


def _extract_names(prompt: str, items: set[str]) -> list[str]:
    names = []
    for word in re.findall(r"\b[A-Z][a-z]+\b", prompt):
        lowered = word.lower()
        if lowered not in IGNORED_NAMES and lowered not in items and word not in names:
            names.append(word)
    return names


def _parse_constraints(prompt: str, names: list[str], items: list[str]):
    constraints = []
    questions = []
    for sentence in re.split(r"[.?!]", prompt):
        lowered = sentence.strip().lower()
        if not lowered or not RELATION_RE.search(lowered) or ":" in sentence or "(" in sentence:
            continue
        found_items = [item for item in items if re.search(rf"\b{re.escape(item)}\b", lowered)]
        found_names = [name for name in names if re.search(rf"\b{re.escape(name.lower())}\b", lowered)]
        if any(word in lowered for word in ("who", "what", "which")):
            if "who" in lowered and len(found_items) == 1 and not found_names:
                questions.append(("who", found_items[0]))
            elif len(found_names) == 1 and not found_items:
                questions.append(("what", found_names[0]))
            else:
                return None
            continue
        if len(found_names) != 1 or not found_items:
            return None
        negated = bool(re.search(r"\b(not|never)\b|n['’]t", lowered))
        if not negated and len(found_items) != 1:
            return None
        constraints.extend((found_names[0], item, not negated) for item in found_items)
    return (constraints, questions) if constraints and questions else None
