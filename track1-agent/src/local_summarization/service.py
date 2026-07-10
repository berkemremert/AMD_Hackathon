import sys
import re
from src.local_summarization import config, parser, compressor, validator, generator
from src.local_summarization.constraints import LocalSummaryResult, ParsedSummaryRequest

def get_word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

class LocalSummaryService:
    def __init__(self):
        self.config = config.get_config()
        self.generator_instance = None # Lazy initialized
        
    def _init_generator(self):
        if self.generator_instance is None:
            self.generator_instance = generator.get_generator(self.config)

    def summarize(self, prompt: str) -> LocalSummaryResult:
        try:
            # 1. Parse
            parsed_req = parser.parse_summary_request(prompt)
            original_words = get_word_count(parsed_req.source_text)
            
            # 2. Compress (optional)
            compressed_source = compressor.compress_source(parsed_req.source_text, parsed_req.constraints)
            compressed_words = get_word_count(compressed_source)
            compression_applied = compressed_words < original_words
            
            # 3. Generate
            self._init_generator()
            attempt = self.generator_instance.generate(parsed_req, compressed_source)
            
            attempts = [attempt]
            total_latency = attempt.latency_ms
            
            # 4. Validate
            val_result = validator.validate_summary(attempt.output, parsed_req.constraints)
            
            final_output = attempt.output
            
            # 5. Repair Pass
            if not val_result.valid and self.config.maximum_repair_attempts > 0:
                print(f"Validation failed: {val_result.errors}. Triggering repair pass...", file=sys.stderr)
                repair_attempt = self.generator_instance.generate(
                    parsed_req, 
                    compressed_source, 
                    is_repair=True, 
                    previous_output=attempt.output, 
                    validation_errors=val_result.errors
                )
                attempts.append(repair_attempt)
                total_latency += repair_attempt.latency_ms
                
                # Re-validate
                val_result = validator.validate_summary(repair_attempt.output, parsed_req.constraints)
                final_output = repair_attempt.output

            return LocalSummaryResult(
                answer=validator.strip_preamble(final_output),
                success=val_result.valid,
                model_id=self.config.model_id,
                compression_applied=compression_applied,
                original_word_count=original_words,
                compressed_word_count=compressed_words,
                validation=val_result,
                attempts=attempts,
                total_latency_ms=total_latency,
                fallback_reason=None
            )
            
        except Exception as e:
            print(f"Local summarization failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Failure policy handling
            policy = config.get_failure_policy()
            if policy == "fireworks":
                # Will be handled by the caller (agent.py)
                raise
            else:
                return LocalSummaryResult(
                    answer=f"Error: Local summarization failed ({e})",
                    success=False,
                    model_id=self.config.model_id,
                    compression_applied=False,
                    original_word_count=0,
                    compressed_word_count=0,
                    validation=validator.ValidationResult(False, [str(e)], 0, 0, 0, False),
                    attempts=[],
                    total_latency_ms=0.0,
                    fallback_reason=str(e)
                )

# Singleton instance
_service = LocalSummaryService()

def summarize(prompt: str) -> LocalSummaryResult:
    return _service.summarize(prompt)
