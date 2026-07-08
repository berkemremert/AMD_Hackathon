"""Category 6 (Code Debugging) pipeline.

Orchestrates: prompt construction → LLM inference → output cleaning →
syntax validation → optional retry.  Designed as a drop-in module so the
final Track 1 router can call ``debugger.solve(prompt)`` regardless of
the underlying model backend.
"""

from prompts import CAT6_CODE_ONLY_PROMPT, CAT6_BUG_PLUS_CODE_PROMPT, CAT6_RETRY_PROMPT
from output_cleaner import clean_model_output
from validators import detect_language, validate_code


class Category6Debugger:
    """Two-pass code-debugging agent for Category 6 tasks.

    Args:
        llm: Any object exposing a ``generate(prompt: str) -> str`` method.
        mode: ``"code_only"`` returns just corrected code;
              ``"bug_plus_code"`` returns a one-line bug description + code.
    """

    def __init__(self, llm, mode: str = "code_only"):
        self.llm = llm
        self.mode = mode

    def solve(self, prompt: str) -> str:
        """Fix a buggy code snippet described by *prompt*.

        Detects the programming language, builds an appropriate prompt,
        queries the LLM, cleans the output, and validates syntax.  If
        validation fails, retries once with a stricter prompt.

        Returns:
            The best cleaned answer (retry result if valid, else first attempt).
        """
        language = detect_language(prompt)

        if self.mode == "bug_plus_code":
            full_prompt = CAT6_BUG_PLUS_CODE_PROMPT.format(prompt=prompt)
            code_only = False
        else:
            full_prompt = CAT6_CODE_ONLY_PROMPT.format(prompt=prompt)
            code_only = True

        raw_answer = self.llm.generate(full_prompt)
        cleaned = clean_model_output(raw_answer, code_only=code_only)

        is_valid, error = validate_code(cleaned, language)

        if is_valid:
            return cleaned

        retry_prompt = CAT6_RETRY_PROMPT.format(
            prompt=prompt,
            bad_answer=cleaned,
            error=error,
        )
        retry_raw = self.llm.generate(retry_prompt)
        retry_cleaned = clean_model_output(retry_raw, code_only=code_only)

        retry_valid, _ = validate_code(retry_cleaned, language)

        if retry_valid:
            return retry_cleaned

        return cleaned
