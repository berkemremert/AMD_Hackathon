"""
Local Solvers Module
Handles specific tasks entirely on the local CPU to achieve a 0-token cost.
"""
from __future__ import annotations
import json
import warnings
import re
import sys
from typing import Optional

# Suppress HuggingFace/Torch warnings for cleaner output
warnings.filterwarnings("ignore")

def extract_target_text(prompt: str) -> str:
    """Isolate the actual text to be processed, stripping away the instructional prompt."""
    if "Sentence:" in prompt:
        return prompt.split("Sentence:")[-1].split("\n\n")[0].strip()
    elif "Text:" in prompt:
        return prompt.split("Text:")[-1].split("\n\n")[0].strip()
    elif '"' in prompt or "'" in prompt:
        import re
        match = re.search(r'["\']([^"\']*)["\']', prompt)
        if match:
            return match.group(1).strip()
    return prompt

def solve_code_debug(prompt: str) -> Optional[str]:
    """
    Disabled local solver for code debugging to avoid 10-minute timeouts on 2 vCPU servers.
    Returns None to cleanly fall back to the Fireworks API.
    """
    return None

def solve_code_authoring(prompt: str) -> Optional[str]:
    """
    Disabled local solver for code authoring to avoid 10-minute timeouts on 2 vCPU servers.
    Returns None to cleanly fall back to the Fireworks API.
    """
    return None

LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "DATE": "DATE",
}

MONTHS = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December"
)

DATE_PATTERN = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,?\s+\d{{4}})?\b",
    re.IGNORECASE,
)

def run_local_ner(text: str, labels: list[str]) -> list[dict]:
    try:
        from src.local_ner.gliner_solver import _load_gliner
        model = _load_gliner()
        if model:
            # restore threshold to 0.3 to avoid common noun false positives
            return model.predict_entities(text, labels, threshold=0.3)
    except Exception as e:
        import sys
        print(f"[WARN] run_local_ner failed: {e}", file=sys.stderr)
    return []

def normalize_model_entities(model_entities: list[dict]) -> list[dict]:
    entities = []
    for ent in model_entities:
        lbl = str(ent.get("label", "")).upper()
        norm_label = LABEL_MAP.get(lbl, lbl)
        entities.append({
            "text": ent["text"],
            "label": norm_label,
            "start": ent["start"],
            "end": ent["end"],
        })
    return entities

def deduplicate_exact_entities(entities: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for ent in entities:
        key = (ent["text"], ent["label"], ent["start"], ent["end"])
        if key not in seen:
            seen.add(key)
            result.append(ent)
    return result

def validate_ner_result(text: str, entities: list[dict]) -> bool:
    allowed = {"PERSON", "ORGANIZATION", "LOCATION", "DATE"}

    if not entities:
        return False

    for entity in entities:
        if entity["label"] not in allowed:
            return False
        if entity["text"] not in text:
            return False

    expected_dates = {m.group(0) for m in DATE_PATTERN.finditer(text)}
    returned_dates = {
        entity["text"]
        for entity in entities
        if entity["label"] == "DATE"
    }

    if expected_dates - returned_dates:
        return False

    return True

def solve_ner(prompt: str) -> Optional[str]:
    text = extract_target_text(prompt)

    model_entities = run_local_ner(
        text,
        labels=["person", "organization", "location"],
    )

    entities = normalize_model_entities(model_entities)

    for match in DATE_PATTERN.finditer(text):
        entities.append({
            "text": match.group(0),
            "label": "DATE",
            "start": match.start(),
            "end": match.end(),
        })

    entities = deduplicate_exact_entities(entities)
    entities.sort(key=lambda item: item["start"])

    if not validate_ner_result(text, entities):
        return None

    return "\n".join(
        f'{entity["text"]} — {entity["label"]}'
        for entity in entities
    )

_vader_sia = None

def solve_sentiment(prompt: str) -> Optional[str]:
    text = extract_target_text(prompt)
    if not text:
        return None
        
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        import sys
        print("[WARN] vaderSentiment not installed, skipping local sentiment.", file=sys.stderr)
        return None
        
    global _vader_sia
    if _vader_sia is None:
        _vader_sia = SentimentIntensityAnalyzer()
    sia = _vader_sia
    
    lower_text = text.lower()
    for cw in [" but ", " yet ", " although ", " though ", " however ", " despite "]:
        if cw in lower_text:
            idx = lower_text.find(cw)
            part1 = text[:idx].strip().strip("',.")
            part2 = text[idx + len(cw):].strip().strip("',.")
            
            score1 = sia.polarity_scores(part1)['compound']
            score2 = sia.polarity_scores(part2)['compound']
            
            if (score1 >= 0.1 and score2 <= -0.1) or (score1 <= -0.1 and score2 >= 0.1):
                return f"Neutral — the review acknowledges both that {part1}, and that {part2}."

    overall_score = sia.polarity_scores(text)['compound']
    
    if overall_score >= 0.55:
        return "Positive — the reviewer expresses clear satisfaction and praises the subject."
    if overall_score <= -0.55:
        return "Negative — the reviewer expresses clear dissatisfaction and criticizes the subject."
        
    if -0.1 <= overall_score <= 0.1:
        # Protect against VADER missing tech support complaints
        for neg_cue in ["crash", "flicker", "drain", "replied", "loose", "scratched", "confusing"]:
            if neg_cue in lower_text:
                return None
        return "Neutral — the statement provides objective, factual information without expressing approval or dissatisfaction."

    return None

import ast
from itertools import permutations

def solve_math_exact(prompt: str) -> Optional[str]:
    # Remove common prefix phrases
    clean_prompt = re.sub(r"^(what's|what is|calculate|evaluate|solve|compute|how much is)\b[:,]?\s*", "", prompt, flags=re.IGNORECASE).strip()
    clean_prompt = clean_prompt.replace('$', '').replace(',', '').replace('x', '*').replace('×', '*').replace('÷', '/').rstrip('?.= ')
    
    # Check if only contains safe math chars
    if not clean_prompt or not re.match(r"^[0-9+\-*/(). ]+$", clean_prompt) or len(clean_prompt) > 100:
        return None
    
    # Must have an operator, otherwise it's just a number
    if not re.search(r"[+\-*/]", clean_prompt):
        return None
        
    try:
        # Extra safety check using ast
        parsed = ast.parse(clean_prompt, mode='eval')
        if not isinstance(parsed, ast.Expression):
            return None
        
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
                return None # No variables or functions allowed
                
        result = eval(compile(parsed, filename="<ast>", mode="eval"), {"__builtins__": None}, {})
        
        if not isinstance(result, (int, float)) or isinstance(result, bool):
            return None
            
        # Format output
        if abs(result - round(result)) < 1e-9 and abs(result) < 1e15:
            return str(int(round(result)))
        return f"{result:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return None


def solve_math_pal(prompt: str) -> Optional[str]:
    return None  # Disabled to save RAM

def solve_logic_puzzle(prompt: str) -> Optional[str]:
    import re
    
    # Extract domain (e.g. colon list, parens, or curly braces)
    domain_match = re.search(r":\s*([a-z]+(?:[ ,]+[a-z]+)+)\s*(?:[.?!]|$)", prompt, re.IGNORECASE)
    if not domain_match:
        domain_match = re.search(r"\(\s*([a-z]+(?:\s*,\s*[a-z]+)+[^)]*)\)", prompt, re.IGNORECASE)
    if not domain_match:
        domain_match = re.search(r"\{([a-z]+(?:\s*,\s*[a-z]+)+[^}]*)\}", prompt, re.IGNORECASE)
    
    if not domain_match:
        return None
        
    items = [word.lower() for word in re.findall(r"[a-zA-Z]+", domain_match.group(1)) if word.lower() not in {"a", "an", "the", "and", "or"}]
    
    if len(items) < 2 or len(items) > 6 or len(set(items)) != len(items):
        return None
        
    item_set = set(items)
    
    ignore_words = {"the", "a", "an", "who", "what", "which", "when", "where", "why", "how", "if", "each", "every", "no", "none", "one", "two", "three", "four", "five", "both", "neither", "either", "he", "she", "it", "they", "we", "you", "i", "and", "or", "but", "so", "then", "also", "here", "there", "this", "that", "these", "those", "their", "his", "her", "its", "there", "three", "friends"}
    
    names = []
    seen_names = set()
    for word in re.findall(r"\b[A-Z][a-z]+\b", prompt):
        lw = word.lower()
        if lw not in ignore_words and lw not in item_set and lw not in seen_names:
            seen_names.add(lw)
            names.append(word)
            
    if len(names) != len(items):
        return None
        
    constraints = []
    questions = []
    
    for sentence in re.split(r"[.?!]", prompt):
        s_clean = sentence.strip().lower()
        if not s_clean or not re.search(r"\b(owns?|have|has|possess(?:es)?|purchased?|bought|choose|chose|gets?|likes?)\b", s_clean):
            continue
            
        if ":" in sentence or "(" in sentence:
            continue
            
        found_items = [i for i in items if re.search(rf"\b{re.escape(i)}\b", s_clean)]
        found_names = [n for n in names if re.search(rf"\b{re.escape(n.lower())}\b", s_clean)]
        
        is_query = "who" in s_clean or "what" in s_clean or "which" in s_clean
        
        if is_query:
            if "who" in s_clean and len(found_items) == 1 and len(found_names) == 0:
                questions.append(("who", found_items[0]))
            elif ("what" in s_clean or "which" in s_clean) and len(found_names) == 1 and len(found_items) == 0:
                questions.append(("what", found_names[0]))
            else:
                return None
            continue
            
        if len(found_names) != 1 or len(found_items) == 0:
            return None
            
        is_negated = bool(re.search(r"\b(not|never)\b|n['’]t", s_clean))
        
        person = found_names[0]
        if is_negated:
            for it in found_items:
                constraints.append((person, it, False)) # False means does not own
        else:
            if len(found_items) != 1:
                return None
            constraints.append((person, found_items[0], True)) # True means owns
            
    if not constraints or not questions:
        return None
        
    valid_answers = None
    
    for p in permutations(items):
        mapping = dict(zip(names, p))
        reverse_mapping = dict(zip(p, names))
        
        valid = True
        for person, item, should_own in constraints:
            if (mapping[person] == item) != should_own:
                valid = False
                break
                
        if valid:
            current_ans = []
            for q_type, target in questions:
                if q_type == "who":
                    current_ans.append(f"{reverse_mapping[target]} owns the {target}")
                else:
                    current_ans.append(f"{target} owns the {mapping[target]}")
            
            if valid_answers is None:
                valid_answers = current_ans
            elif valid_answers != current_ans:
                return None # Multiple solutions exist
                
    if not valid_answers:
        return None
        
    return ", and ".join(valid_answers) + "."


def solve_summarization(prompt: str) -> Optional[str]:
    """
    Disabled summarization solver to save RAM.
    """
    return None

