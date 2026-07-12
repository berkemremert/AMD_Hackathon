"""Conservative local three-way sentiment classification."""
from __future__ import annotations

import sys
from typing import Any

from .common import extract_target_text


_analyzer: Any | None = None
CONTRASTS = (" but ", " yet ", " although ", " though ", " however ", " despite ")
UNCERTAIN_NEGATIVE_CUES = (
    "crash",
    "flicker",
    "drain",
    "replied",
    "loose",
    "scratched",
    "confusing",
)


def _get_analyzer() -> Any | None:
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    except ImportError:
        print("[WARN] vaderSentiment is unavailable.", file=sys.stderr)
    return _analyzer


def solve_sentiment(prompt: str) -> str | None:
    text = extract_target_text(prompt)
    analyzer = _get_analyzer()
    if not text or analyzer is None:
        return None

    lower_text = text.lower()
    for contrast in CONTRASTS:
        if contrast not in lower_text:
            continue
        index = lower_text.find(contrast)
        first = text[:index].strip().strip("',.")
        second = text[index + len(contrast) :].strip().strip("',.")
        first_score = analyzer.polarity_scores(first)["compound"]
        second_score = analyzer.polarity_scores(second)["compound"]
        if (first_score >= 0.1 and second_score <= -0.1) or (
            first_score <= -0.1 and second_score >= 0.1
        ):
            return f"Neutral — the review acknowledges both that {first}, and that {second}."

    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.55:
        return "Positive — the reviewer expresses clear satisfaction and praises the subject."
    if score <= -0.55:
        return "Negative — the reviewer expresses clear dissatisfaction and criticizes the subject."
    if -0.1 <= score <= 0.1:
        if any(cue in lower_text for cue in UNCERTAIN_NEGATIVE_CUES):
            return None
        return (
            "Neutral — the statement provides objective, factual information without "
            "expressing approval or dissatisfaction."
        )
    return None
