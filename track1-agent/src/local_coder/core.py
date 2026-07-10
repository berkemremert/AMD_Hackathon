"""
Core Implementation for Local Neural Coder (`Qwen/Qwen2.5-Coder-1.5B-Instruct`).
Provides thread-safe, lazy-loaded local inference for code debugging and code generation tasks at 0 API tokens.
If local model execution fails, returns None to smoothly trigger the Tier 3 API fallback.
"""
import sys
import threading
import re
import ast
from typing import Optional

_MODEL = None
_TOKENIZER = None
_LOCK = threading.Lock()
_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER
    
    with _LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"[LOCAL CODER] Loading local model '{_MODEL_ID}' onto memory (0 API tokens)...", file=sys.stderr)
            tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID,
                dtype=dtype,
                low_cpu_mem_usage=True
            ).to(device)
            model.eval()
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
            _MODEL = model
            _TOKENIZER = tokenizer
            print(f"[LOCAL CODER] '{_MODEL_ID}' loaded successfully on {device}.", file=sys.stderr)
            return _MODEL, _TOKENIZER
        except Exception as e:
            print(f"[LOCAL CODER] Failed to load local model '{_MODEL_ID}': {e}", file=sys.stderr)
            return None, None

def solve_qwen_coder(prompt: str, task_type: str = "code_debugging") -> Optional[str]:
    """
    Runs Qwen2.5-Coder locally (Tier 2) for general code debugging or code authoring.
    Returns the formatted string if valid, or None to fall back to the API (Tier 3).
    """
    model, tokenizer = _load_model()
    if model is None or tokenizer is None:
        return None

    try:
        import torch
        device = model.device
        if task_type == "code_debugging" or "identify the bug" in prompt.lower():
            system_prompt = "State the bug briefly, then provide the corrected code in a single fenced block. Be concise."
            max_tokens = 160
        else:
            system_prompt = "Output only the code in a single fenced block — correct, complete, and self-contained. No preamble."
            max_tokens = 320

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Verify AST correctness if code block is present
        blocks = re.findall(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if not blocks:
            # Check for generic fenced block
            blocks = re.findall(r"```\s*(.*?)\s*```", response, re.DOTALL)
            
        if blocks:
            code = blocks[-1].strip()
            try:
                ast.parse(code)
                return response
            except SyntaxError as e:
                print(f"[LOCAL CODER] Qwen generated invalid Python syntax ({e.msg}). Falling back to API...", file=sys.stderr)
                return None
        elif task_type == "code_authoring":
            # If code authoring generated text without code block, wrap or check syntax
            try:
                ast.parse(response)
                return f"```python\n{response}\n```"
            except SyntaxError:
                return None
                
        return response
    except Exception as e:
        print(f"[LOCAL CODER] Execution error during Qwen inference ({e}). Falling back to API...", file=sys.stderr)
        return None
