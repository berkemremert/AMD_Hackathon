"""Shared prompt parsing helpers for local solvers."""
import re


def extract_target_text(prompt: str) -> str:
    """Extract the quoted or explicitly labelled text from a task prompt."""
    for marker in ("Sentence:", "Text:"):
        if marker in prompt:
            return prompt.split(marker, 1)[1].split("\n\n", 1)[0].strip()

    quoted = re.search(r'["\']([^"\']*)["\']', prompt)
    return quoted.group(1).strip() if quoted else prompt
