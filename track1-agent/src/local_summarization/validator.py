import re
import json
from src.local_summarization.constraints import SummaryConstraints, ValidationResult

def count_words(text: str) -> int:
    # Basic word boundary counting
    return len(re.findall(r'\b\w+\b', text))

def count_sentences(text: str) -> int:
    # Naive sentence splitting (similar to nltk sent_tokenize but deterministic and fast)
    # Match punctuation followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return len([s for s in sentences if s.strip()])

def count_bullets(text: str) -> int:
    return len(re.findall(r'(?m)^[ \t]*[-*•\d]+\.?\s+', text))

def validate_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except:
        return False

def strip_preamble(text: str) -> str:
    """Removes obvious 'Summary:', 'Here is a summary:', etc."""
    text = re.sub(r'^(here is a summary.*?:\s*|summary:\s*|the summary is:\s*)', '', text, flags=re.IGNORECASE).strip()
    # Remove surrounding quotes if they wrap the entire text
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text

def detect_repetition(text: str) -> bool:
    """Fails validation if the same sentence is repeated exactly."""
    sentences = [s.strip().lower() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return len(sentences) != len(set(sentences))

def validate_summary(output: str, constraints: SummaryConstraints) -> ValidationResult:
    errors = []
    
    clean_output = strip_preamble(output)
    if not clean_output:
        return ValidationResult(False, ["Empty output"], 0, 0, 0, False)
        
    word_count = count_words(clean_output)
    sent_count = count_sentences(clean_output)
    bullet_count = count_bullets(clean_output)
    
    # Check words
    if constraints.exact_words is not None:
        if word_count != constraints.exact_words:
            errors.append(f"Expected exactly {constraints.exact_words} words, got {word_count}")
    if constraints.max_words is not None:
        if word_count > constraints.max_words:
            errors.append(f"Expected max {constraints.max_words} words, got {word_count}")
            
    # Check sentences
    if constraints.exact_sentences is not None:
        if sent_count != constraints.exact_sentences:
            errors.append(f"Expected exactly {constraints.exact_sentences} sentences, got {sent_count}")
    if constraints.max_sentences is not None:
        if sent_count > constraints.max_sentences:
            errors.append(f"Expected max {constraints.max_sentences} sentences, got {sent_count}")
            
    # Check bullets
    if constraints.bullet_count is not None:
        if bullet_count != constraints.bullet_count:
            errors.append(f"Expected exactly {constraints.bullet_count} bullets, got {bullet_count}")
            
    # Format checks
    format_valid = True
    if constraints.output_format == "json":
        format_valid = validate_json(clean_output)
        if not format_valid:
            errors.append("Invalid JSON format")
            
    if detect_repetition(clean_output):
        errors.append("Output contains repeated sentences")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        word_count=word_count,
        sentence_count=sent_count,
        bullet_count=bullet_count,
        format_valid=format_valid
    )
