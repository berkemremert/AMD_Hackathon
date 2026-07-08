"""Prompt templates for the Category 6 code-debugging agent.

Three templates are defined:
  - ``CAT6_CODE_ONLY_PROMPT``:     Asks for corrected code only (default).
  - ``CAT6_BUG_PLUS_CODE_PROMPT``: Asks for a bug description + corrected code.
  - ``CAT6_RETRY_PROMPT``:         Stricter follow-up when the first answer had
                                   syntax errors.

All templates accept ``{prompt}`` (the original task).  The retry template
additionally accepts ``{bad_answer}`` and ``{error}``.
"""

CAT6_CODE_ONLY_PROMPT = """You are a code debugging agent.

Your job:
Fix the bug in the provided code.

Rules:
- Output only the corrected code.
- Do not include explanations.
- Do not include markdown fences.
- Preserve the original structure as much as possible.
- Make the smallest correct change.
- If the bug involves concurrency, synchronization, mutation, indexing, parsing, type handling, async behavior, resource handling, or validation, fix it directly in code.
- If the original code includes imports, class names, function names, or an if __name__ == "__main__" block, preserve them unless they are part of the bug.
- Do not invent unrelated functionality.

Task prompt:
{prompt}
"""

CAT6_BUG_PLUS_CODE_PROMPT = """You are a code debugging agent.

Your job:
Identify the bug and provide the corrected implementation.

Output format:
Bug: one concise sentence.
Fix:
<corrected code>

Rules:
- Keep the bug explanation to one sentence.
- Do not include markdown fences.
- Preserve the original structure as much as possible.
- Make the smallest correct change.
- Do not invent unrelated functionality.

Task prompt:
{prompt}
"""

CAT6_RETRY_PROMPT = """Your previous answer had a syntax problem.

Fix the code again.

Rules:
- Output only valid corrected code.
- Do not explain.
- Do not include markdown fences.
- Preserve the original code structure.
- Make the minimal fix required.
- Ensure the code is syntactically valid.

Original task:
{prompt}

Previous invalid answer:
{bad_answer}

Validation error:
{error}
"""
