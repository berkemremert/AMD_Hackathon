import re
from decimal import Decimal

def classify_task(prompt: str) -> str:
    """Classifies the prompt into one of the 8 categories using regex."""
    p = prompt.lower()
    
    # 6. code_debugging: looking for bugs, errors, fix, debug, traceback
    if re.search(r'\b(bug|debug|fix|error|traceback|exception)\b', p) and re.search(r'\b(code|function|script|snippet)\b', p):
        return "code_debugging"
        
    # 8. code_generation: write, implement, function, algorithm in code
    if re.search(r'\b(write a function|implement a|create a python script|write a script|code a)\b', p):
        return "code_generation"
        
    # 2. mathematical reasoning: numbers, calculate, percentages, average
    if re.search(r'\b(calculate|what is|how many|percentage|sum|average|difference|multiply|divide|cost|total)\b', p) and re.search(r'\d+', prompt):
        return "math"
        
    # 3. sentiment classification: positive, negative, sentiment, review, feedback
    if re.search(r'\b(sentiment|positive|negative|neutral|classify this review)\b', p):
        return "sentiment"
        
    # 4. text summarisation: summarize, tl;dr, condense, short summary
    if re.search(r'\b(summarize|summarise|tl;dr|condense|brief summary)\b', p):
        return "summarization"
        
    # 5. named entity recognition: entities, NER, names, organizations, locations, dates
    if re.search(r'\b(extract|identify)\b.*\b(entities|names|organizations|locations|dates|ner)\b', p):
        return "ner"
        
    # 7. logical / deductive reasoning: logic puzzle, constraints, deductive, knights and knaves
    if re.search(r'\b(puzzle|logical|deductive|constraints|who is the|which one is|must be true)\b', p) and re.search(r'\b(if|then|and|or|not)\b', p):
        return "logical_reasoning"
        
    # 1. factual knowledge: fallback or direct factual cues
    return "factual_knowledge"

def maybe_solve_locally(category: str, prompt: str):
    """Attempt to solve basic deterministic math locally to save 100% of tokens."""
    if category != "math":
        return None
        
    # E.g., "What is 15% of 200?"
    perc_match = re.search(r'what is (\d+(?:\.\d+)?)%\s*of\s*(\d+(?:\.\d+)?)', prompt.lower())
    if perc_match:
        try:
            percent = Decimal(perc_match.group(1))
            val = Decimal(perc_match.group(2))
            res = (percent / Decimal(100)) * val
            return '{0:f}'.format(res.normalize()) # Removes trailing zeros and avoids scientific notation
        except:
            pass
            
    return None

def build_minimal_prompt(category: str, original_prompt: str) -> str:
    """Wraps the prompt with instructions designed to minimize token output."""
    base_instructions = {
        "code_debugging": "Output ONLY the corrected code snippet. No explanations.",
        "code_generation": "Output ONLY the requested code snippet. No markdown, no explanations.",
        "math": "Output ONLY the final numerical answer. No steps, no units unless requested.",
        "sentiment": "Output ONLY the sentiment label (e.g., Positive, Negative, Neutral).",
        "summarization": "Provide an extremely concise summary. Do not include intro or outro text.",
        "ner": "Output ONLY a JSON array of extracted entities. No extra text.",
        "logical_reasoning": "Output ONLY the final deduced answer in one sentence.",
        "factual_knowledge": "Answer in exactly one concise sentence. No conversational filler."
    }
    
    instruction = base_instructions.get(category, "Answer directly and concisely. No filler text.")
    return f"{instruction}\n\nTask: {original_prompt}"

def post_process(category: str, text: str) -> str:
    """Cleans up the LLM output."""
    t = text.strip()
    
    # Strip markdown if code
    if category in ["code_generation", "code_debugging"]:
        t = re.sub(r'^```[\w]*\n', '', t)
        t = re.sub(r'\n```$', '', t)
        t = t.strip()
        
    # Remove conversational filler
    fillers = [
        "Sure, here is", "Here is", "The answer is", "Here's", 
        "Certainly!", "Sure thing", "Based on the text,"
    ]
    for f in fillers:
        if t.lower().startswith(f.lower()):
            t = t[len(f):].lstrip(' :,-')
            
    return t.strip()
