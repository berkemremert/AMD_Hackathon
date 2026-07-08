Below is a **copy-paste-ready instruction brief** you can give to your LLM coding agent/team.

---

# Instructions for LLM Agent Team: Category 6 Code Debugging Local Model Pipeline

We are participating in **Track 1: General-Purpose / Hybrid Token-Efficient Routing Agent** for the AMD Developer Hackathon. The full Track 1 system will eventually need to handle all 8 categories, but our current focus is **Category 6: Code Debugging**.

Category 6 is defined as: **identifying bugs in code snippets and providing corrected implementations**. The Track 1 container is expected to read tasks from `/input/tasks.json` and write answers to `/output/results.json`. 

The goal of this task is to build a **local model-based Category 6 debugging module** that can run locally now and later be integrated into the final agent pipeline.

---

## 1. Main objective

Build a Python-based local debugging agent for Category 6.

The agent should:

1. Load a JSON file containing tasks.
2. Detect or assume Category 6 code-debugging tasks.
3. Send the prompt to a local code-specialized model.
4. Receive a corrected implementation.
5. Clean the model output.
6. Validate syntax when possible.
7. Retry once if the output is invalid.
8. Save results in the expected format.

The output format must be:

```json
[
  {
    "task_id": "cat6_synthetic_001",
    "answer": "corrected code or concise fix answer"
  }
]
```

The final Track 1 guide says `/output/results.json` must be valid JSON, and malformed output scores zero. It also says submissions are ranked by token efficiency after passing an accuracy gate. 

---

## 2. Important rule note

There is a rule ambiguity we must respect.

The participant guide says local models and local tokens count as zero, but also says all inference must go through Fireworks AI via `FIREWORKS_BASE_URL`.  The event page says local models are optional and useful for development/testing, while only Fireworks-routed inference is counted in final scoring. 

Therefore:

For now, implement the local model path for development and experimentation.

For final submission, keep the code modular so we can switch between:

```text
LOCAL_MODEL_MODE=true
```

and

```text
FIREWORKS_MODE=true
```

Do **not** hardcode answers. Do **not** cache answers. Evaluation uses unseen prompt variants. 

---

## 3. Target local model

Use this model first:

```text
qwen2.5-coder:3b
```

Runner:

```text
Ollama
```

Development machine:

```text
Mac M1, 8GB RAM
```

Install and run:

```bash
brew install ollama
ollama serve
ollama pull qwen2.5-coder:3b
ollama run qwen2.5-coder:3b
```

The local model should be called through Ollama’s local HTTP API:

```text
http://localhost:11434/api/generate
```

---

## 4. Expected repository structure

Create this structure:

```text
agent/
  main.py
  cat6_debugger.py
  local_llm.py
  output_cleaner.py
  validators.py
  prompts.py
  config.py
  eval_cat6.py
  requirements.txt
data/
  cat6_tasks.json
outputs/
  cat6_local_results.json
```

Responsibilities:

```text
main.py
```

Entry point. Loads tasks, routes each task, writes results.

```text
cat6_debugger.py
```

Category 6 pipeline: prompt construction, local model call, cleaning, validation, retry.

```text
local_llm.py
```

Ollama client.

```text
output_cleaner.py
```

Removes markdown fences, explanations, duplicate text, and keeps only the useful answer.

```text
validators.py
```

Language detection and syntax validation.

```text
prompts.py
```

Stores prompt templates.

```text
eval_cat6.py
```

Runs local evaluation over the 100 Category 6 examples.

---

## 5. Category 6 processing flow

Implement the flow exactly like this:

```text
Input task
  ↓
Extract task_id and prompt
  ↓
Detect language if possible
  ↓
Build Category 6 debugging prompt
  ↓
Call local Qwen2.5-Coder-3B through Ollama
  ↓
Clean model output
  ↓
Validate syntax if language is supported
  ↓
If invalid, retry once with stricter prompt
  ↓
If still invalid, return best cleaned answer
  ↓
Append {task_id, answer} to results
```

For now, do not over-engineer routing. If the input task has:

```json
"category": "code_debugging"
```

route it directly to Category 6.

If there is no category field, route to Category 6 when the prompt contains terms like:

```text
bug
debug
fix
corrected implementation
code snippet
script
function
error
stack trace
exception
```

---

## 6. Primary prompt template

Use this as the first prompt template.

```python
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
```

This is the default mode because the user currently wants **only corrected code**.

---

## 7. Alternative prompt template for accuracy testing

Some Category 6 prompts may ask to “identify the bug” or “explain the fix.” Since the judge is LLM-based, a concise explanation may score better in some cases.

Implement a second mode:

```python
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
```

The evaluation script should test both modes:

```text
mode = "code_only"
mode = "bug_plus_code"
```

Then compare which format performs better on the 100 local Category 6 examples.

---

## 8. Ollama client implementation

Create `local_llm.py`:

```python
import requests


class LocalLLM:
    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 0.9,
        num_ctx: int = 4096,
        num_predict: int = 900,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
```

Use:

```text
temperature = 0
```

for stable debugging output.

---

## 9. Output cleaning rules

Create `output_cleaner.py`.

The cleaner must:

1. Remove leading/trailing whitespace.
2. Remove markdown fences like:

````text
```python
...
````

````

3. If the model returns multiple code blocks, keep the largest code block.
4. If the prompt mode is `code_only`, remove explanation before or after the code when possible.
5. Preserve valid code indentation.
6. Do not remove imports.
7. Do not remove `if __name__ == "__main__"` blocks unless the model duplicated them.

Suggested implementation:

```python
import re


def extract_code_blocks(text: str) -> list[str]:
    pattern = r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```"
    return re.findall(pattern, text, flags=re.DOTALL)


def clean_model_output(text: str, code_only: bool = True) -> str:
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
````

---

## 10. Language detection

Create a simple detector in `validators.py`.

````python
import re


def detect_language(prompt: str) -> str:
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
````

Do not rely too heavily on language detection because the user said the dataset may not explicitly state the language.

---

## 11. Syntax validation

Implement lightweight validation only when easy.

For Python:

```python
import ast


def validate_python(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
```

For JavaScript:

```python
import subprocess
import tempfile
from pathlib import Path


def validate_javascript(code: str) -> tuple[bool, str]:
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
```

For unsupported languages:

```python
def validate_code(code: str, language: str) -> tuple[bool, str]:
    if language == "python":
        return validate_python(code)
    if language == "javascript":
        return validate_javascript(code)

    # Unknown language: do not fail just because we cannot validate.
    return True, ""
```

Validation should not execute untrusted code. Syntax checking is acceptable; running arbitrary code is risky and unnecessary for now.

---

## 12. Retry prompt

If syntax validation fails, retry once with a stricter prompt.

```python
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
```

Retry only once to avoid wasting time.

---

## 13. Category 6 debugger module

Create `cat6_debugger.py`.

```python
from prompts import CAT6_CODE_ONLY_PROMPT, CAT6_BUG_PLUS_CODE_PROMPT, CAT6_RETRY_PROMPT
from output_cleaner import clean_model_output
from validators import detect_language, validate_code


class Category6Debugger:
    def __init__(self, llm, mode: str = "code_only"):
        self.llm = llm
        self.mode = mode

    def solve(self, prompt: str) -> str:
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
```

---

## 14. Main script

Create `main.py`.

For local testing, read from:

```text
data/cat6_tasks.json
```

For final integration, also support:

```text
/input/tasks.json
```

Implementation:

```python
import json
from pathlib import Path

from local_llm import LocalLLM
from cat6_debugger import Category6Debugger


def load_tasks() -> list[dict]:
    official_path = Path("/input/tasks.json")
    local_path = Path("data/cat6_tasks.json")

    if official_path.exists():
        return json.loads(official_path.read_text())

    if local_path.exists():
        return json.loads(local_path.read_text())

    raise FileNotFoundError("No tasks file found.")


def write_results(results: list[dict]) -> None:
    official_output_dir = Path("/output")
    local_output_dir = Path("outputs")

    if official_output_dir.exists():
        output_path = official_output_dir / "results.json"
    else:
        local_output_dir.mkdir(exist_ok=True)
        output_path = local_output_dir / "cat6_local_results.json"

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    tasks = load_tasks()

    llm = LocalLLM(model="qwen2.5-coder:3b")
    debugger = Category6Debugger(llm=llm, mode="code_only")

    results = []

    for index, task in enumerate(tasks, start=1):
        task_id = task["task_id"]
        prompt = task["prompt"]

        print(f"[{index}/{len(tasks)}] Solving {task_id}")

        answer = debugger.solve(prompt)

        results.append({
            "task_id": task_id,
            "answer": answer,
        })

    write_results(results)


if __name__ == "__main__":
    main()
```

---

## 15. Evaluation script

Create `eval_cat6.py`.

This script should:

1. Load the 100 Category 6 examples.
2. Run mode `code_only`.
3. Run mode `bug_plus_code`.
4. Save both outputs.
5. Report basic metrics.

Metrics to report:

```text
total_tasks
avg_answer_length
python_syntax_valid_count
empty_answer_count
markdown_fence_count
contains_explanation_count
```

This does not replace LLM judging, but it helps quickly catch bad behavior.

````python
import json
from pathlib import Path

from local_llm import LocalLLM
from cat6_debugger import Category6Debugger
from validators import detect_language, validate_code


def evaluate_mode(tasks: list[dict], mode: str) -> list[dict]:
    llm = LocalLLM(model="qwen2.5-coder:3b")
    debugger = Category6Debugger(llm=llm, mode=mode)

    results = []

    for i, task in enumerate(tasks, start=1):
        print(f"[{mode}] [{i}/{len(tasks)}] {task['task_id']}")
        answer = debugger.solve(task["prompt"])
        language = detect_language(task["prompt"])
        is_valid, error = validate_code(answer, language)

        results.append({
            "task_id": task["task_id"],
            "answer": answer,
            "language": language,
            "syntax_valid": is_valid,
            "syntax_error": error,
            "answer_length": len(answer),
        })

    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    return {
        "total_tasks": total,
        "syntax_valid_count": sum(1 for r in results if r["syntax_valid"]),
        "empty_answer_count": sum(1 for r in results if not r["answer"].strip()),
        "markdown_fence_count": sum(1 for r in results if "```" in r["answer"]),
        "avg_answer_length": sum(r["answer_length"] for r in results) / max(total, 1),
    }


def main():
    tasks = json.loads(Path("data/cat6_tasks.json").read_text())

    Path("outputs").mkdir(exist_ok=True)

    for mode in ["code_only", "bug_plus_code"]:
        results = evaluate_mode(tasks, mode)
        summary = summarize(results)

        Path(f"outputs/cat6_{mode}_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2)
        )

        Path(f"outputs/cat6_{mode}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )

        print(mode, summary)


if __name__ == "__main__":
    main()
````

---

## 16. Requirements file

Create `requirements.txt`:

```text
requests
```

Do not add heavy dependencies yet.

---

## 17. Local run instructions

Run Ollama:

```bash
ollama serve
```

Pull the model:

```bash
ollama pull qwen2.5-coder:3b
```

Install Python dependencies:

```bash
pip install -r agent/requirements.txt
```

Run the local agent:

```bash
python agent/main.py
```

Run evaluation:

```bash
python agent/eval_cat6.py
```

Expected output files:

```text
outputs/cat6_local_results.json
outputs/cat6_code_only_results.json
outputs/cat6_code_only_summary.json
outputs/cat6_bug_plus_code_results.json
outputs/cat6_bug_plus_code_summary.json
```

---

## 18. Quality requirements

The agent should satisfy these before we integrate it into the full Track 1 system:

1. It completes all 100 Category 6 examples without crashing.
2. It produces valid JSON output.
3. It does not include markdown fences in final answers.
4. It does not output empty answers.
5. For Python tasks, at least 90% of outputs should pass `ast.parse`.
6. It preserves original function/class names whenever possible.
7. It makes minimal changes.
8. It avoids long explanations in `code_only` mode.
9. Runtime should be reasonable on Mac M1 with 8GB RAM.
10. The code should be modular enough to replace Ollama with Fireworks later.

---

## 19. Common bug types to handle well

The prompt template and model behavior should be tested against these patterns:

```text
Race condition / missing lock
Off-by-one loop error
Mutable default argument
Incorrect async / await usage
Wrong exception handling
Incorrect file handling
Resource leak
Wrong variable name
Incorrect recursion base case
Integer division vs float division
Incorrect sorting key
Wrong regex
Incorrect SQL query
Bad date parsing
Incorrect type conversion
Missing return statement
Shared state mutation
Wrong comparison operator
Incorrect boundary condition
```

The threading example should produce a fix using `threading.Lock`.

---

## 20. Final integration note

For final Track 1, build this as a module, not a standalone-only script.

The final system should eventually look like:

```text
main.py
  ↓
router.py
  ↓
category handlers:
  cat1_factual.py
  cat2_math.py
  cat3_sentiment.py
  cat4_summary.py
  cat5_ner.py
  cat6_debugger.py
  cat7_logic.py
  cat8_codegen.py
```

Category 6 should expose one clean method:

```python
answer = category6_debugger.solve(prompt)
```

That way, we can later decide whether `solve()` uses:

```text
local Qwen2.5-Coder
```

or:

```text
Fireworks allowed model
```

without rewriting the router.

---

## 21. Do not do these things

Do not hardcode answers.

Do not rely on the 100 local examples being the real test set.

Do not cache prompt-answer pairs.

Do not output malformed JSON.

Do not include `.env` values in the final image.

Do not hardcode Fireworks model IDs in final mode; the guide says they must come from `ALLOWED_MODELS`. 

Do not call Fireworks outside `FIREWORKS_BASE_URL` in final mode; the guide says calls bypassing that URL will not be recorded. 

Do not spend time building a complex UI or API server right now. This task is batch input → batch output.

---

## 22. Immediate deliverables

The LLM agent team should deliver:

```text
1. Working local Category 6 pipeline using Ollama + qwen2.5-coder:3b
2. main.py that reads tasks and writes results
3. eval_cat6.py that compares code_only vs bug_plus_code modes
4. Syntax validation for Python and JavaScript
5. Output cleaner that removes markdown fences
6. Summary JSON showing local evaluation metrics
7. Clean modular code ready for future Fireworks integration
```

The first successful milestone is:

```text
Given data/cat6_tasks.json with 100 code-debugging tasks,
running python agent/eval_cat6.py produces two result files and two summary files without crashing.
```
