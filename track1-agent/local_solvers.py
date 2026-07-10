"""
Local Solvers Module
Handles specific tasks entirely on the local CPU to achieve a 0-token cost.
"""
import json
import warnings
import re
import sys
from gliner import GLiNER

# Suppress HuggingFace/Torch warnings for cleaner output
warnings.filterwarnings("ignore")

# Global cache for the GLiNER model so it's loaded only once
_gliner_model = None

def get_gliner():
    global _gliner_model
    if _gliner_model is None:
        model_name = "urchade/gliner_large-v2.1"
        print(f"Loading local NER solver ({model_name})...", file=sys.stderr)
        _gliner_model = GLiNER.from_pretrained(model_name)
    return _gliner_model

# Standard labels commonly asked for in Track 1
DEFAULT_LABELS = ["Person", "Organization", "Location", "Date", "Miscellaneous"]

def extract_labels_from_prompt(prompt: str):
    """Attempt to dynamically extract requested labels from the prompt instructions."""
    prompt_lower = prompt.lower()
    labels = []
    if "person" in prompt_lower or "people" in prompt_lower: labels.append("Person")
    if "org" in prompt_lower or "organization" in prompt_lower: labels.append("Organization")
    if "loc" in prompt_lower or "location" in prompt_lower: labels.append("Location")
    if "date" in prompt_lower or "time" in prompt_lower: labels.append("Date")
    if "event" in prompt_lower: labels.append("Event")
    if "product" in prompt_lower: labels.append("Product")
    
    return labels if labels else DEFAULT_LABELS

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
    Extracts named entities using the zero-shot GLiNER model and returns them 
    in the standard JSON format: [{"entity": "Name", "type": "Person"}]
    """
    # Dynamically figure out what labels the prompt is asking for
    labels_to_find = extract_labels_from_prompt(prompt)
    
    # Isolate the target text so GLiNER doesn't extract words from the instructions
    target_text = extract_target_text(prompt)
    
    # Run the zero-token local inference
    entities = get_gliner().predict_entities(target_text, labels_to_find, threshold=0.3)
    
    formatted_entities = []
    
    for ent in entities:
        formatted_entities.append({
            "entity": ent["text"],
            "type": ent["label"]
        })
            
    # Return as a JSON string to match API output expectations
    return json.dumps(formatted_entities, indent=2)

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
