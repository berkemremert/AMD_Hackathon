"""GLiNER plus deterministic date extraction."""
from __future__ import annotations

import re
import sys

from ..gliner import load_model
from .common import extract_target_text


LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
}
MONTHS = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
)
DATE_PATTERN = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b",
    re.IGNORECASE,
)
ALLOWED_LABELS = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}


def _model_entities(text: str) -> list[dict]:
    try:
        model = load_model()
        if model is None:
            return []
        return model.predict_entities(
            text,
            ["person", "organization", "location"],
            threshold=0.3,
        )
    except Exception as exc:
        print(f"[WARN] Local NER failed: {exc}", file=sys.stderr)
        return []


def _normalize(entities: list[dict]) -> list[dict]:
    normalized = []
    for entity in entities:
        label = LABEL_MAP.get(str(entity.get("label", "")).upper(), "")
        normalized.append(
            {
                "text": entity["text"],
                "label": label,
                "start": entity["start"],
                "end": entity["end"],
            }
        )
    return normalized


def _deduplicate(entities: list[dict]) -> list[dict]:
    unique = {}
    for entity in entities:
        key = (entity["text"], entity["label"], entity["start"], entity["end"])
        unique[key] = entity
    return sorted(unique.values(), key=lambda entity: entity["start"])


def _is_valid(text: str, entities: list[dict]) -> bool:
    if not entities:
        return False
    if any(entity["label"] not in ALLOWED_LABELS for entity in entities):
        return False
    if any(entity["text"] not in text for entity in entities):
        return False

    expected_dates = {match.group(0) for match in DATE_PATTERN.finditer(text)}
    returned_dates = {
        entity["text"] for entity in entities if entity["label"] == "DATE"
    }
    return expected_dates <= returned_dates


def solve_ner(prompt: str) -> str | None:
    text = extract_target_text(prompt)
    entities = _normalize(_model_entities(text))
    entities.extend(
        {
            "text": match.group(0),
            "label": "DATE",
            "start": match.start(),
            "end": match.end(),
        }
        for match in DATE_PATTERN.finditer(text)
    )
    entities = _deduplicate(entities)
    if not _is_valid(text, entities):
        return None
    return "\n".join(f'{entity["text"]} — {entity["label"]}' for entity in entities)
