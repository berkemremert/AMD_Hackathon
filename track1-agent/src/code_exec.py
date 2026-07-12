"""Runs LLM-generated Python functions against known test cases in a subprocess.

Used only for the code_generation category, where we have concrete input/output
pairs and can grade objectively instead of relying on an LLM judge. This runs
arbitrary model-generated code — fine for this project's own controlled dev-time
labeling, but never do this with untrusted user input in a real service without
a real sandbox (container, gVisor, nsjail, etc.).
"""
import re
import subprocess
import sys
import tempfile

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer_text: str) -> str:
    match = CODE_BLOCK_RE.search(answer_text)
    if match:
        return match.group(1)
    return answer_text


def run_tests(answer_text: str, function_name: str, tests: list, timeout: int = 10) -> bool:
    """Returns True only if the function is defined and every test passes."""
    code = extract_code(answer_text)
    harness = (
        code
        + "\n\nimport sys\n"
        + f"_tests = {tests!r}\n"
        + f"_fn = {function_name}\n"
        + "_all_ok = True\n"
        + "for _t in _tests:\n"
        + "    try:\n"
        + "        _got = _fn(*_t['args'])\n"
        + "        if _got != _t['expected']:\n"
        + "            _all_ok = False\n"
        + "    except Exception:\n"
        + "        _all_ok = False\n"
        + "print('PASS' if _all_ok else 'FAIL')\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip().endswith("PASS")
    except Exception:
        return False
