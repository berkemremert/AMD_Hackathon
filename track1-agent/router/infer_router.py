"""Fast, dependency-free difficulty router.

The original router loaded a 255 MB DistilBERT checkpoint. Besides adding
startup and memory cost, a damaged checkpoint made the production agent fail
before it could call Fireworks. Its training target also treated every failure
of the efficient model as "hard", even when the escalation model failed too.

This router retains the legacy easy/hard interface for compatibility. Model
selection currently maps both labels to Kimi, while the category detection is
still used for local solvers and output constraints.

``predict`` keeps the old public interface and never raises for ordinary text.
"""
from __future__ import annotations

import re

from output_optimizer import detect_task_type


_STRICT_CONSTRAINT_RE = re.compile(
    r"\b(?:must|ensure|strictly|exactly|forbidden|do not|without using|"
    r"edge cases?|all bugs?|every bug|production[- ]ready|thread[- ]safe|"
    r"race conditions?|deadlocks?|memory leaks?)\b",
    re.IGNORECASE,
)
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
_BULLET_ITEM_RE = re.compile(r"(?m)^\s*[-*•]\s+")
_ADVANCED_CODE_RE = re.compile(
    r"\b(?:concurren|thread[- ]safe|async|deadlock|race condition|lock[- ]free|"
    r"distributed|dynamic programming|graph algorithm|parser|compiler|"
    r"production[- ]ready|exception safety|memory safety|generic|template)\b",
    re.IGNORECASE,
)


def _constraint_count(prompt: str) -> int:
    """Approximate the number of independently judgeable requirements."""
    return (
        len(_STRICT_CONSTRAINT_RE.findall(prompt))
        + len(_NUMBERED_ITEM_RE.findall(prompt))
        + len(_BULLET_ITEM_RE.findall(prompt))
    )


def predict(prompt: str) -> str:
    """Return ``easy`` or ``hard`` using conservative complexity signals."""
    if not isinstance(prompt, str) or not prompt.strip():
        return "easy"

    task_type = detect_task_type(prompt)
    word_count = len(prompt.split())
    constraints = _constraint_count(prompt)
    code_size = sum(len(block) for block in re.findall(r"```[^\n]*\n(.*?)```", prompt, re.DOTALL))

    if task_type == "bug_fixing":
        # On the labeled benchmark, escalation on the 50 substantial debugging
        # tasks recovered four answers and used fewer tokens overall.
        if word_count >= 110 or code_size >= 700:
            return "hard"
        if constraints >= 6 and _ADVANCED_CODE_RE.search(prompt):
            return "hard"

    elif task_type == "code_authoring":
        # Longer generation specifications generally require a real algorithm
        # rather than a one-expression utility function. Historical results
        # show a net five-answer gain and lower tokens at this boundary.
        if word_count >= 50 or _ADVANCED_CODE_RE.search(prompt):
            return "hard"

    return "easy"


if __name__ == "__main__":
    import sys

    print(predict(sys.argv[1] if len(sys.argv) > 1 else "What is 2 + 2?"))
