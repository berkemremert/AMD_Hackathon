"""
GLiNER-based NER Solver
Uses urchade/gliner_small-v2.1 (~600MB) for accurate zero-shot named entity recognition.
Replaces the heuristic pipeline while reusing format detection and output formatting.
"""
import sys
import threading
from typing import List, Optional

from src.local_ner.core import (
    Entity,
    parse_target_text,
    detect_format,
    extract_dates,
    resolve_overlaps,
    deduplicate_entities,
    format_output,
)

_GLINER_MODEL = None
_GLINER_LOCK = threading.Lock()
_MODEL_ID = "urchade/gliner_small-v2.1"

# The entity labels GLiNER will search for (mapped to our internal label scheme)
_GLINER_LABELS = ["person", "organization", "location"]
_LABEL_MAP = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "LOCATION",
}


def _load_gliner():
    global _GLINER_MODEL
    if _GLINER_MODEL is not None:
        return _GLINER_MODEL

    with _GLINER_LOCK:
        if _GLINER_MODEL is not None:
            return _GLINER_MODEL
        try:
            from gliner import GLiNER

            print(f"[GLiNER] Loading {_MODEL_ID}...", file=sys.stderr)
            model = GLiNER.from_pretrained(_MODEL_ID)
            _GLINER_MODEL = model
            print(f"[GLiNER] {_MODEL_ID} loaded successfully.", file=sys.stderr)
            return _GLINER_MODEL
        except Exception as e:
            print(f"[GLiNER] Failed to load model: {e}", file=sys.stderr)
            return None


def extract_entities_gliner(text: str, threshold: float = 0.3) -> List[Entity]:
    """Run GLiNER inference on the text and return Entity objects."""
    model = _load_gliner()
    if model is None:
        return []

    raw_entities = model.predict_entities(text, _GLINER_LABELS, threshold=threshold)

    entities = []
    for ent in raw_entities:
        label = _LABEL_MAP.get(ent["label"], ent["label"].upper())
        entities.append(
            Entity(
                text=ent["text"],
                label=label,
                start=ent["start"],
                end=ent["end"],
                score=ent["score"],
                source="gliner",
            )
        )
    return entities


def solve_ner_gliner(prompt: str) -> Optional[str]:
    """Full NER pipeline using GLiNER + regex dates + existing formatting."""
    target_text = parse_target_text(prompt)
    instruction_text = prompt.replace(target_text, "")
    if not instruction_text.strip():
        instruction_text = prompt

    format_type = detect_format(instruction_text)

    # 1. Regex dates (reliable, keep as-is)
    date_entities = extract_dates(target_text)

    # 2. GLiNER for person/org/location
    gliner_entities = extract_entities_gliner(target_text)

    # 3. Merge, resolve overlaps, dedup
    all_candidates = date_entities + gliner_entities
    resolved = resolve_overlaps(all_candidates)
    deduped = deduplicate_entities(resolved)

    if not deduped:
        return None

    return format_output(deduped, format_type)
