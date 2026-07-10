"""
Local Extractive Compression Pipeline for Summarization, Bug Fixing, and QA.
Uses all-MiniLM-L6-v2 for semantic extraction and Regex for code minification.
"""

import re
import threading
import sys
from dataclasses import dataclass, field
from typing import Optional, Any
import torch

@dataclass
class SummaryConstraints:
    max_words: Optional[int] = None
    exact_words: Optional[int] = None
    sentence_count: Optional[int] = None
    bullet_count: Optional[int] = None
    output_format: Optional[str] = None
    one_sentence: bool = False

@dataclass
class ParsedSummaryRequest:
    instruction: str
    source_text: str
    constraints: SummaryConstraints
    parsing_confidence: float

@dataclass
class CompressionConfig:
    lambda_relevance: float = 0.75
    duplicate_threshold: float = 0.84
    bonus_number: float = 0.05
    bonus_percent: float = 0.05
    bonus_date: float = 0.05
    bonus_proper_name: float = 0.05
    bonus_first_sentence: float = 0.10
    bonus_conclusion: float = 0.05
    penalty_short: float = 0.20
    penalty_boilerplate: float = 0.30

@dataclass
class CompressionResult:
    compressed_text: str
    original_word_count: int
    compressed_word_count: int
    original_sentence_count: int
    selected_sentence_count: int
    compression_ratio: float
    applied: bool
    fallback_reason: Optional[str] = None
    selected_indices: list[int] = field(default_factory=list)

_model_cache: dict = {}
_model_lock = threading.Lock()
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def _load_embedding_model() -> tuple[Any, str]:
    if "loaded" in _model_cache:
        return _model_cache["loaded"]

    with _model_lock:
        if "loaded" in _model_cache:
            return _model_cache["loaded"]

        import os
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("LOCAL_EMBEDDING_DEVICE", "cpu")
        try:
            model = SentenceTransformer(_MODEL_NAME, device=device)
            _model_cache["loaded"] = (model, device)
        except Exception as e:
            raise RuntimeError(f"Failed to load {_MODEL_NAME}: {e}")

    return _model_cache["loaded"]

def parse_summarization_prompt(prompt: str) -> ParsedSummaryRequest:
    """Safely split instruction and source text."""
    instruction = prompt
    source = ""
    confidence = 0.0

    # 1. Fenced code/text block
    block_match = re.search(r"^(.*?)(```(?:text)?\n(.*?)\n```)", prompt, re.DOTALL | re.IGNORECASE)
    if block_match:
        instruction = block_match.group(1).strip()
        source = block_match.group(3).strip()
        confidence = 0.9
    else:
        # 2. Explicit markers
        marker_match = re.search(r"^(.*?)\b(Text|Passage|Article|Content|Source):\s*\n(.*)$", prompt, re.DOTALL | re.IGNORECASE)
        if marker_match:
            instruction = marker_match.group(1).strip()
            source = marker_match.group(3).strip()
            confidence = 0.8
        else:
            # 3. Phrases
            phrase_match = re.search(r"^(.*?(?:Summarize the following text|Summarise the following passage|Condense the following|Summarize the following)[:\s]+)(.*)$", prompt, re.DOTALL | re.IGNORECASE)
            if phrase_match:
                instruction = phrase_match.group(1).strip()
                source = phrase_match.group(2).strip()
                confidence = 0.7
            else:
                # 4. Fallback: just split by double newline and assume last part is source if it's long
                parts = prompt.split("\n\n", 1)
                if len(parts) == 2 and len(parts[1]) > len(parts[0]):
                    instruction = parts[0].strip()
                    source = parts[1].strip()
                    confidence = 0.5
                else:
                    source = prompt
                    confidence = 0.1

    constraints = _extract_constraints(instruction)
    return ParsedSummaryRequest(instruction, source, constraints, confidence)

def _extract_constraints(instruction: str) -> SummaryConstraints:
    ins_lower = instruction.lower()
    constraints = SummaryConstraints()
    
    if "one sentence" in ins_lower or "single sentence" in ins_lower:
        constraints.one_sentence = True
        
    match_sent = re.search(r"(?:in|exactly) (\d+) sentences", ins_lower)
    if match_sent:
        constraints.sentence_count = int(match_sent.group(1))
        
    match_bull = re.search(r"(\d+) bullet points", ins_lower)
    if match_bull:
        constraints.bullet_count = int(match_bull.group(1))
        
    match_words = re.search(r"(?:at most|no more than|under|maximum|exactly) (\d+) words", ins_lower)
    if match_words:
        if "exactly" in ins_lower:
            constraints.exact_words = int(match_words.group(1))
        else:
            constraints.max_words = int(match_words.group(1))
            
    if "json" in ins_lower:
        constraints.output_format = "json"
    elif "table" in ins_lower:
        constraints.output_format = "table"
        
    return constraints

@dataclass
class SentenceMeta:
    index: int
    text: str
    words: int
    has_number: bool
    has_date: bool
    has_percent: bool
    has_proper_name: bool
    is_first: bool
    is_conclusion: bool
    starts_with_discourse: bool

def segment_sentences(text: str) -> list[SentenceMeta]:
    """Conservative English sentence splitter."""
    import re
    # Split on . ? ! followed by space and capital letter, but avoid decimals/initials/titles.
    pattern = r"(?<!\b[A-Z])(?<!\b[A-Z]\.)(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bVs\.)(?<!\bProf\.)(?<!\bSt\.)(?<=[.!?])\s+(?=[A-Z])|\n\n"
    raw_sentences = re.split(pattern, text)
    
    meta_list = []
    for i, s in enumerate(raw_sentences):
        s_clean = s.strip()
        if not s_clean:
            continue
        words = len(s_clean.split())
        s_lower = s_clean.lower()
        
        has_number = bool(re.search(r"\d", s_clean))
        has_percent = bool(re.search(r"\d\s*%", s_clean)) or "percent" in s_lower
        has_date = bool(re.search(r"\b(19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}\b", s_clean))
        has_proper_name = bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", s_clean))
        is_first = (i == 0)
        is_conclusion = "in conclusion" in s_lower or "to summarize" in s_lower or "overall" in s_lower
        starts_with_discourse = bool(re.match(r"^(however|therefore|consequently|this|these|those|it|they|such|which|as a result)\b", s_lower))
        
        meta_list.append(SentenceMeta(i, s_clean, words, has_number, has_date, has_percent, has_proper_name, is_first, is_conclusion, starts_with_discourse))
        
    return meta_list

def calculate_budget(original_words: int, constraints: SummaryConstraints) -> int:
    """Calculate the target compressed word count."""
    if constraints.one_sentence or (constraints.max_words and constraints.max_words <= 50):
        return min(250, max(60, int(original_words * 0.15)))
    elif constraints.sentence_count or constraints.bullet_count or (constraints.max_words and constraints.max_words <= 100):
        return min(350, max(80, int(original_words * 0.2)))
    else:
        return min(450, max(100, int(original_words * 0.25)))

def repair_cohesion(selected_indices: list[int], sentences: list[SentenceMeta], budget: int) -> list[int]:
    final_indices = set(selected_indices)
    current_words = sum(sentences[i].words for i in final_indices)
    
    for idx in selected_indices:
        if sentences[idx].starts_with_discourse and idx > 0:
            if idx - 1 not in final_indices:
                if current_words + sentences[idx-1].words <= budget + 50:
                    final_indices.add(idx - 1)
                    current_words += sentences[idx-1].words
                    
    return sorted(list(final_indices))

def select_sentences_mmr(sentences: list[SentenceMeta], config: CompressionConfig, budget: int) -> list[int]:
    try:
        model, device = _load_embedding_model()
    except Exception as e:
        return [s.index for s in sentences]

    texts = [s.text for s in sentences]
    
    with torch.inference_mode():
        embeddings = model.encode(texts, convert_to_tensor=True, device=device, normalize_embeddings=True)
        centroid = torch.mean(embeddings, dim=0, keepdim=True)
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
        
        relevance_scores = torch.nn.functional.cosine_similarity(embeddings, centroid).cpu().numpy()
        
    for i, s in enumerate(sentences):
        if s.has_number: relevance_scores[i] += config.bonus_number
        if s.has_percent: relevance_scores[i] += config.bonus_percent
        if s.has_date: relevance_scores[i] += config.bonus_date
        if s.has_proper_name: relevance_scores[i] += config.bonus_proper_name
        if s.is_first: relevance_scores[i] += config.bonus_first_sentence
        if s.is_conclusion: relevance_scores[i] += config.bonus_conclusion
        if s.words < 5: relevance_scores[i] -= config.penalty_short
        
    selected = []
    unselected = list(range(len(sentences)))
    current_words = 0
    
    while unselected and current_words < budget:
        mmr_scores = []
        for idx in unselected:
            if not selected:
                mmr_scores.append(relevance_scores[idx])
            else:
                sims = torch.nn.functional.cosine_similarity(embeddings[idx].unsqueeze(0), embeddings[selected])
                max_sim = torch.max(sims).item()
                if max_sim > config.duplicate_threshold:
                    mmr_scores.append(-1.0)
                else:
                    score = config.lambda_relevance * relevance_scores[idx] - (1 - config.lambda_relevance) * max_sim
                    mmr_scores.append(score)
                    
        best_idx = unselected[max(range(len(mmr_scores)), key=mmr_scores.__getitem__)]
        
        if mmr_scores[unselected.index(best_idx)] == -1.0:
            break
            
        selected.append(best_idx)
        current_words += sentences[best_idx].words
        unselected.remove(best_idx)
        
    return sorted(selected)

def compress_summarization_prompt(user_prompt: str, config: Optional[CompressionConfig] = None) -> str:
    if config is None:
        config = CompressionConfig()
        
    try:
        parsed = parse_summarization_prompt(user_prompt)
        
        if parsed.parsing_confidence < 0.6:
            return user_prompt
            
        original_words = len(parsed.source_text.split())
        
        if original_words < 100:
            return user_prompt
            
        sentences = segment_sentences(parsed.source_text)
        if len(sentences) < 6:
            return user_prompt
            
        budget = calculate_budget(original_words, parsed.constraints)
        
        selected_indices = select_sentences_mmr(sentences, config, budget)
        selected_indices = repair_cohesion(selected_indices, sentences, budget)
        
        compressed_text = " ".join([sentences[i].text for i in selected_indices])
        
        return f"{parsed.instruction}\n\n{compressed_text}"
        
    except Exception as e:
        print(f"[Compressor] Summarization Error: {e}", file=sys.stderr)
        return user_prompt

def compress_code_debugging(user_prompt: str) -> str:
    """Removes docstrings, comments, and empty lines from code blocks."""
    def minifier(match):
        code = match.group(2)
        # Remove block strings (single and double quotes)
        code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue # remove empty lines
            if stripped.startswith('#') or stripped.startswith('//'):
                continue # remove full-line comments
            new_lines.append(line)
        return match.group(1) + "\n" + "\n".join(new_lines) + "\n```"
        
    # Match ```python ... ``` or ``` ... ```
    new_prompt = re.sub(r"(```[a-zA-Z]*\n)(.*?)(```)", minifier, user_prompt, flags=re.DOTALL)
    
    # Also remove extra empty lines in the prompt itself
    new_prompt = re.sub(r'\n{3,}', '\n\n', new_prompt)
    return new_prompt

def compress_knowledge_qa(user_prompt: str) -> str:
    """Uses RAG to extract only sentences relevant to the question."""
    parts = user_prompt.split("\n\n")
    if len(parts) < 2:
        return user_prompt 
        
    question_idx = -1
    for i, p in enumerate(parts):
        if '?' in p:
            if question_idx == -1 or len(p) < len(parts[question_idx]):
                question_idx = i
                
    if question_idx == -1:
        question_idx = 0
        
    question_text = parts[question_idx]
    
    context_parts = [p for i, p in enumerate(parts) if i != question_idx]
    context_text = "\n\n".join(context_parts)
    
    if len(context_text.split()) < 50:
        return user_prompt 
        
    sentences = segment_sentences(context_text)
    if len(sentences) < 5:
        return user_prompt
        
    try:
        model, device = _load_embedding_model()
    except Exception:
        return user_prompt
        
    try:
        with torch.inference_mode():
            q_emb = model.encode([question_text], convert_to_tensor=True, device=device, normalize_embeddings=True)
            s_embs = model.encode([s.text for s in sentences], convert_to_tensor=True, device=device, normalize_embeddings=True)
            
            sims = torch.nn.functional.cosine_similarity(q_emb, s_embs).cpu().numpy()
            
        budget = max(3, int(len(sentences) * 0.4))
        top_indices = sims.argsort()[-budget:][::-1]
        top_indices = sorted(list(top_indices)) 
        
        compressed_context = " ".join([sentences[i].text for i in top_indices])
        
        if question_idx == 0:
            return f"{question_text}\n\n[Extracted Context]: {compressed_context}"
        else:
            return f"[Extracted Context]: {compressed_context}\n\n{question_text}"
            
    except Exception as e:
        print(f"[Compressor] QA Error: {e}", file=sys.stderr)
        return user_prompt


def optimize_prompt_for_api(user_prompt: str, task_type: str, suffix: str = "") -> str:
    """Universal Compression Dispatcher."""
    compressed = user_prompt
    
    if task_type == "summarization":
        compressed = compress_summarization_prompt(user_prompt)
    elif task_type in ["code_debugging", "bug_fixing", "code_authoring"]:
        compressed = compress_code_debugging(user_prompt)
        # Aggressively remove explanation demands
        compressed = re.sub(r"(?i)Explain\s+what\s+(the\s+)?bug\s+is\s+and\s+show\s+the\s+corrected\s+code\.?", "", compressed)
        compressed = re.sub(r"(?i)Explain\s+the\s+bug\s+and\s+give\s+the\s+fixed\s+code\.?", "", compressed)
        compressed = re.sub(r"(?i)Explain\s+what\s+(the\s+)?bug\s+is\s+and\s+how\s+(your\s+)?fix\s+resolves\s+it\.?", "", compressed)
        compressed = re.sub(r"(?i)Provide\s+an\s+explanation\.?", "", compressed)
    elif task_type in ["factual_knowledge", "knowledge_qa"]:
        compressed = compress_knowledge_qa(user_prompt)
    elif task_type in ["math_solving", "logical_puzzles"]:
        compressed = re.sub(r"(?i)Explain\s+your\s+reasoning\.?", "", compressed)
        compressed = re.sub(r"(?i)Show\s+your\s+work\.?", "", compressed)
        
    # The suffix contains output instructions required by the API, so append it safely
    if suffix:
        return f"{suffix}\n\n{compressed}"
    return compressed
