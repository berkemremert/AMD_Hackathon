from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SummaryConstraints:
    max_words: Optional[int] = None
    exact_words: Optional[int] = None
    max_sentences: Optional[int] = None
    exact_sentences: Optional[int] = None
    bullet_count: Optional[int] = None
    output_format: Optional[str] = None
    preserve_names: bool = False
    preserve_numbers: bool = False
    focus_instruction: Optional[str] = None

@dataclass
class ParsedSummaryRequest:
    original_prompt: str
    instruction: str
    source_text: str
    constraints: SummaryConstraints
    parsing_confidence: float

@dataclass
class GenerationAttempt:
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempt_type: str

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    word_count: int
    sentence_count: int
    bullet_count: int
    format_valid: bool

@dataclass
class LocalSummaryResult:
    answer: str
    success: bool
    model_id: str
    compression_applied: bool
    original_word_count: int
    compressed_word_count: int
    validation: ValidationResult
    attempts: List[GenerationAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0
    fallback_reason: Optional[str] = None
