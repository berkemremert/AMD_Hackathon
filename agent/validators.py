"""Language detection and syntax validation utilities.

Provides lightweight heuristics to guess the programming language of a
task prompt and syntax-check the generated answer before returning it.
"""

import re
import ast
import subprocess
import tempfile
from pathlib import Path


def detect_language(prompt: str) -> str:
    """Guess the programming language from keywords in *prompt*.

    Uses simple substring/regex matching.  Returns ``"unknown"`` if
    no language can be confidently identified.
    """
    lower = prompt.lower()

    if "```python" in lower or "python" in lower:
        return "python"
    if "```javascript" in lower or "javascript" in lower or "node.js" in lower:
        return "javascript"
    if "```typescript" in lower or "typescript" in lower:
        return "typescript"
    if "```java" in lower or re.search(r"\bpublic\s+class\b", prompt):
        return "java"
    if "```cpp" in lower or "c++" in lower or "#include <" in prompt:
        return "cpp"
    if "```c" in lower or "#include <" in prompt:
        return "c"
    if "sql" in lower or "select " in lower:
        return "sql"

    return "unknown"


def validate_python(code: str) -> tuple[bool, str]:
    """Check Python code for syntax errors using ``ast.parse``.

    Returns:
        ``(True, "")`` on success, ``(False, error_message)`` on failure.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def validate_javascript(code: str) -> tuple[bool, str]:
    """Check JavaScript code via ``node --check``.

    Returns:
        ``(True, "")`` on success, ``(False, stderr)`` on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.js"
        path.write_text(code)
        result = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stderr


def validate_code(code: str, language: str) -> tuple[bool, str]:
    """Dispatch syntax validation to the appropriate language checker.

    For unsupported languages, returns ``(True, "")`` to avoid false negatives.
    """
    if language == "python":
        return validate_python(code)
    if language == "javascript":
        return validate_javascript(code)

    # Unknown language: do not fail just because we cannot validate.
    return True, ""
