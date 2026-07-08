"""LLM output post-processing.

Strips markdown fences, explanation prefixes, and other boilerplate
that local models tend to add even when instructed not to, so the
final answer contains only clean code.
"""

import re


def extract_code_blocks(text: str) -> list[str]:
    """Extract all fenced code blocks from markdown-formatted text.

    Returns:
        A list of code strings found inside triple-backtick fences.
    """
    pattern = r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```"
    return re.findall(pattern, text, flags=re.DOTALL)


def clean_model_output(text: str, code_only: bool = True) -> str:
    """Clean raw LLM output into a usable code answer.

    Steps:
      1. If fenced code blocks are present, keep only the largest.
      2. Strip any remaining markdown fence markers.
      3. If *code_only*, remove common explanation prefixes.

    Args:
        text: Raw model output string.
        code_only: When True, aggressively strip explanation text.

    Returns:
        Cleaned answer string.
    """
    text = text.strip()

    code_blocks = extract_code_blocks(text)
    if code_blocks:
        text = max(code_blocks, key=len).strip()

    text = text.replace("```python", "")
    text = text.replace("```javascript", "")
    text = text.replace("```typescript", "")
    text = text.replace("```java", "")
    text = text.replace("```cpp", "")
    text = text.replace("```c++", "")
    text = text.replace("```", "")
    text = text.strip()

    if code_only:
        # Remove common explanation prefixes if the model ignored instructions.
        prefixes = [
            "Here is the corrected code:",
            "Corrected code:",
            "The corrected code is:",
            "Fix:",
        ]
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

    return text
