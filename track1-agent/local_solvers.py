"""
Local Solvers Module
Handles specific tasks entirely on the local CPU to achieve a 0-token cost.
"""
import json
import warnings
import re
import sys
import sys

# Suppress HuggingFace/Torch warnings for cleaner output
warnings.filterwarnings("ignore")

def extract_target_text(prompt: str) -> str:
    """Isolate the actual text to be processed, stripping away the instructional prompt."""
    if "Sentence:" in prompt:
        return prompt.split("Sentence:")[-1].split("\n\n")[0].strip()
    elif "Text:" in prompt:
        return prompt.split("Text:")[-1].split("\n\n")[0].strip()
    elif '"' in prompt:
        import re
        match = re.search(r'"([^"]*)"', prompt)
        if match:
            return match.group(1).strip()
    return prompt

def solve_ner(prompt: str) -> str:
    """
    Extracts named entities using deterministic pipeline and formats them as requested.
    """
    from src.local_ner.core import solve_ner_pipeline
    return solve_ner_pipeline(prompt)

import threading

# Sentiment global cache
_sentiment_pipeline = None
_sentiment_lock = threading.Lock()

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading local sentiment solver ({model_name})...", file=sys.stderr)
        _sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    return _sentiment_pipeline

# Predefined cue words for fake justifications
POSITIVE_CUES = ["love", "fast", "incredibly", "easy", "delicious", "wonderful", "friendly", "great", "excellent", "good"]
NEGATIVE_CUES = ["bad", "buggy", "frustrating", "terrible", "slow", "crashes", "worst", "awful", "hate", "poor"]

def solve_sentiment(prompt: str) -> str:
    """
    Extracts sentiment using a zero-shot/fine-tuned local model and generates a fake justification.
    """
    target_text = extract_target_text(prompt)
    
    with _sentiment_lock:
        pipe = get_sentiment_pipeline()
        result = pipe(target_text)[0]
        
    label = result["label"].capitalize() # "Positive", "Negative", "Neutral"
    
    # Generate justification based on cue words
    text_lower = target_text.lower()
    found_cues = []
    
    if label == "Positive":
        found_cues = [word for word in POSITIVE_CUES if word in text_lower]
        polarity = "positive"
    elif label == "Negative":
        found_cues = [word for word in NEGATIVE_CUES if word in text_lower]
        polarity = "negative"
    else:
        polarity = "neutral"
        
    if found_cues:
        # Use up to 2 cues
        cues_str = " and ".join(f"'{c}'" for c in found_cues[:2])
        justification = f"The text uses {polarity} cues such as {cues_str}."
    else:
        justification = f"The overall tone and language of the text leans {polarity}."
        
    return f"{label}. {justification}"

import ast
from itertools import permutations

def solve_math_exact(prompt: str) -> str | None:
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

def solve_logic_puzzle(prompt: str) -> str | None:
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

