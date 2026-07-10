import re
import math
import sys
import os
from src.local_summarization.config import get_min_compression_words
from src.local_summarization.constraints import SummaryConstraints

# Add root directory to path to import legacy compressor
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import local_summary_compressor as legacy_compressor

def should_compress(source: str, constraints: SummaryConstraints) -> bool:
    """Decides if compression is necessary based on length and config."""
    word_count = len(re.findall(r'\b\w+\b', source))
    min_words = get_min_compression_words()
    
    # Do not compress short passages
    if word_count < min_words:
        return False
        
    return True

def get_budget_multiplier(constraints: SummaryConstraints) -> float:
    """Determine how aggressive compression should be based on constraints."""
    if constraints.exact_words and constraints.exact_words <= 50:
        return 0.15 # Very aggressive
    if constraints.max_words and constraints.max_words <= 50:
        return 0.15
    if constraints.exact_sentences == 1:
        return 0.15
        
    # Standard conservative compression for generic summaries
    return 0.35

def compress_source(source: str, constraints: SummaryConstraints) -> str:
    """Compresses the source text using MiniLM, preserving document order."""
    
    if not should_compress(source, constraints):
        return source
        
    multiplier = get_budget_multiplier(constraints)
    
    try:
        sentences = legacy_compressor.split_into_sentences(source)
        # Fake a prompt context for the compressor, using focus_instruction if available
        # or just a generic summary instruction.
        context_prompt = constraints.focus_instruction if constraints.focus_instruction else "Summarize the text comprehensively."
        
        # We need a model, so we load it lazy style
        model = legacy_compressor.load_embedding_model()
        
        prompt_emb = model.encode([context_prompt])[0]
        sentence_embs = model.encode(sentences)
        
        word_count = sum(len(s.split()) for s in sentences)
        budget = max(50, int(word_count * multiplier))
        
        # Max compression target
        budget = min(budget, 700)
        
        selected_indices = legacy_compressor.select_sentences_mmr(
            prompt_embedding=prompt_emb,
            sentence_embeddings=sentence_embs,
            sentences=sentences,
            budget_words=budget,
            lambda_param=0.6
        )
        
        # Preserve document order
        selected_indices.sort()
        compressed_text = " ".join([sentences[i] for i in selected_indices])
        return compressed_text
    except Exception as e:
        import sys
        print(f"Compression failed, failing open: {e}", file=sys.stderr)
        return source
