"""
Local Solvers Module
Handles specific tasks entirely on the local CPU to achieve a 0-token cost.
"""
import json
import warnings
import re
from gliner import GLiNER

# Suppress HuggingFace/Torch warnings for cleaner output
warnings.filterwarnings("ignore")

print("Loading local NER solver (urchade/gliner_small-v2.1)...")
# We load this globally so it only initializes once.
ner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

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

def solve_ner(prompt: str) -> str:
    """
    Extracts named entities using the zero-shot GLiNER model and returns them 
    in the standard JSON format: [{"entity": "Name", "type": "Person"}]
    """
    # Dynamically figure out what labels the prompt is asking for
    labels_to_find = extract_labels_from_prompt(prompt)
    
    # Run the zero-token local inference
    entities = ner_model.predict_entities(prompt, labels_to_find, threshold=0.3)
    
    formatted_entities = []
    
    for ent in entities:
        formatted_entities.append({
            "entity": ent["text"],
            "type": ent["label"]
        })
            
    # Return as a JSON string to match API output expectations
    return json.dumps(formatted_entities, indent=2)
