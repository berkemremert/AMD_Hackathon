import time
import math
import sys
from typing import Protocol, Dict, Any, Optional
from src.local_summarization.config import LocalGenerationConfig
from src.local_summarization.constraints import ParsedSummaryRequest, GenerationAttempt

class LocalSummaryGenerator(Protocol):
    def generate(self, request: ParsedSummaryRequest, source_text: str, is_repair: bool = False, previous_output: str = "", validation_errors: list = None) -> GenerationAttempt:
        ...

# Global model cache to avoid reloading
_MODEL_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}

class QwenSummaryGenerator(LocalSummaryGenerator):
    def __init__(self, config: LocalGenerationConfig):
        self.config = config
        self._load_model()
        
    def _load_model(self):
        global _MODEL_CACHE, _TOKENIZER_CACHE
        
        if self.config.model_id not in _MODEL_CACHE:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
            except ImportError:
                raise ImportError("transformers and torch must be installed for local generation.")
            
            print(f"Loading local summarization model {self.config.model_id} on {self.config.device}...", file=sys.stderr)
            
            # Determine torch_dtype
            dtype_map = {"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
            torch_dtype = dtype_map.get(self.config.dtype, "auto")
            
            # Lazy load
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, local_files_only=False) # Fallback to download if missing, Docker handles offline
            
            # Use accelerate if needed or standard
            device_map = self.config.device if self.config.device != "cpu" else None
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, 
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=False
            )
            model.eval()
            
            if self.config.device == "cpu" and device_map is None:
                # If explicit CPU is required
                pass
                
            _MODEL_CACHE[self.config.model_id] = model
            _TOKENIZER_CACHE[self.config.model_id] = tokenizer
            print("Local model loaded successfully.", file=sys.stderr)
            
        self.model = _MODEL_CACHE[self.config.model_id]
        self.tokenizer = _TOKENIZER_CACHE[self.config.model_id]

    def _calculate_max_tokens(self, request: ParsedSummaryRequest) -> int:
        c = request.constraints
        if c.exact_words:
            return math.ceil(c.exact_words * 1.8) + 20
        if c.max_words:
            return math.ceil(c.max_words * 1.8) + 20
        if c.exact_sentences == 1:
            return 80
        if c.bullet_count:
            return c.bullet_count * 40 + 20
        return self.config.default_max_new_tokens

    def generate(self, request: ParsedSummaryRequest, source_text: str, is_repair: bool = False, previous_output: str = "", validation_errors: list = None) -> GenerationAttempt:
        import torch
        
        system_prompt = (
            "You produce faithful summaries using only information from the source. "
            "Follow every requested length and format constraint exactly. "
            "Return only the requested summary."
        )
        
        if is_repair:
            errors_str = "\\n- ".join(validation_errors or [])
            user_msg = (
                f"Rewrite the candidate to satisfy all constraints.\n\n"
                f"Constraints:\n- {errors_str}\n- return only the summary\n\n"
                f"Candidate:\n{previous_output}\n\n"
                f"Return only the corrected summary."
            )
        else:
            user_msg = f"Task:\n{request.instruction}\n\nSource:\n{source_text}"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt")
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        max_new_tokens = self._calculate_max_tokens(request)
        
        start_time = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else None,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract output ignoring the prompt
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return GenerationAttempt(
            prompt=prompt,
            output=output_text.strip(),
            input_tokens=inputs["input_ids"].shape[1],
            output_tokens=len(generated_ids[0]),
            latency_ms=latency_ms,
            attempt_type="repair" if is_repair else "initial"
        )

def get_generator(config: LocalGenerationConfig) -> LocalSummaryGenerator:
    # Factory for extending to Flan-T5 in future if needed
    if "qwen" in config.model_id.lower():
        return QwenSummaryGenerator(config)
    # Default to Qwen logic for auto models
    return QwenSummaryGenerator(config)
