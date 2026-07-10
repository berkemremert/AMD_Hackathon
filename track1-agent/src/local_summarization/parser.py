import re
from typing import Tuple
from src.local_summarization.constraints import SummaryConstraints, ParsedSummaryRequest

def split_instruction_and_source(prompt: str) -> Tuple[str, str, float]:
    """Separates the user instruction from the source passage safely."""
    # Common markers
    markers = [
        r"Text:\s*", r"Passage:\s*", r"Article:\s*", 
        r"Content:\s*", r"Source:\s*", r"<source>\s*"
    ]
    
    # Try to find a clear marker
    for marker in markers:
        match = re.search(marker, prompt, re.IGNORECASE)
        if match:
            instruction = prompt[:match.start()].strip()
            source = prompt[match.end():].strip()
            # Remove closing </source> if present
            source = re.sub(r"</source>\s*$", "", source, flags=re.IGNORECASE).strip()
            return instruction, source, 1.0
            
    # If no explicit marker, try to split on multiple newlines assuming first paragraph is instruction
    parts = re.split(r'\n\s*\n', prompt, maxsplit=1)
    if len(parts) == 2 and len(parts[0].split()) < 50:
        return parts[0].strip(), parts[1].strip(), 0.8
        
    # Fallback: treat the whole thing as source if it's very long, but we need an instruction
    # If we can't split safely, we return the prompt as instruction and empty source, 
    # but the logic should handle empty source by generating on the instruction directly.
    return prompt, prompt, 0.3

def parse_summary_request(prompt: str) -> ParsedSummaryRequest:
    instruction, source_text, confidence = split_instruction_and_source(prompt)
    
    constraints = SummaryConstraints()
    
    # Extract exact words
    exact_words_match = re.search(r"exactly\s+(\d+)\s+words?", instruction, re.IGNORECASE)
    if exact_words_match:
        constraints.exact_words = int(exact_words_match.group(1))
        
    # Extract max words
    max_words_match = re.search(r"(?:maximum|no more than|under|at most|less than)\s+(\d+)\s+words?", instruction, re.IGNORECASE)
    if max_words_match:
        constraints.max_words = int(max_words_match.group(1))
        
    # Extract exact sentences
    exact_sents_match = re.search(r"exactly\s+(\d+)\s+sentences?", instruction, re.IGNORECASE)
    if exact_sents_match:
        constraints.exact_sentences = int(exact_sents_match.group(1))
    elif re.search(r"(one sentence|single sentence)", instruction, re.IGNORECASE):
        constraints.exact_sentences = 1
        
    # Extract max sentences
    max_sents_match = re.search(r"(?:maximum|no more than|under|at most|less than)\s+(\d+)\s+sentences?", instruction, re.IGNORECASE)
    if max_sents_match:
        constraints.max_sentences = int(max_sents_match.group(1))
        
    # Extract bullet points
    exact_bullets_match = re.search(r"exactly\s+(\d+)\s+bullet\s+points?", instruction, re.IGNORECASE)
    if exact_bullets_match:
        constraints.bullet_count = int(exact_bullets_match.group(1))
        constraints.output_format = "list"
    else:
        max_bullets_match = re.search(r"(\d+)\s+bullet\s+points?", instruction, re.IGNORECASE)
        if max_bullets_match:
            constraints.bullet_count = int(max_bullets_match.group(1))
            constraints.output_format = "list"
        elif "bullet" in instruction.lower():
            constraints.output_format = "list"
            
    # Formats
    if "json" in instruction.lower():
        constraints.output_format = "json"
        
    # Focus instructions
    focus_match = re.search(r"(focus on|emphasize|include|preserve)\s+([^\.]+)", instruction, re.IGNORECASE)
    if focus_match:
        constraints.focus_instruction = focus_match.group(0).strip()
        
    if "names" in instruction.lower() and "preserve" in instruction.lower():
        constraints.preserve_names = True
    if "numbers" in instruction.lower() and "preserve" in instruction.lower():
        constraints.preserve_numbers = True

    return ParsedSummaryRequest(
        original_prompt=prompt,
        instruction=instruction,
        source_text=source_text if confidence > 0.3 else prompt,
        constraints=constraints,
        parsing_confidence=confidence
    )
