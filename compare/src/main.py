"""
Container entry point for the judging harness.

The harness runs this on startup. It must:
  1. Read tasks from   /input/tasks.json    -> [ {"task_id": "...", "prompt": "..."}, ... ]
  2. Write answers to  /output/results.json -> [ {"task_id": "...", "answer": "..."}, ... ]
  3. Exit with code 0.

Notes:
  - /input and /output are provided by the harness at run time (mounted in).
  - For LOCAL testing you can point these elsewhere with the INPUT_PATH and
    OUTPUT_PATH environment variables (see the README).
"""

import json
import os
import sys

from tokens import TokenLedger
from agent import run_agent


def _load_dotenv():
    """For LOCAL development only: load KEY=VALUE lines from a .env file next to
    the project, if one exists. In the judging container there is no .env, so this
    does nothing and the harness's injected environment variables are used instead.
    """
    for folder in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   os.getcwd()):
        path = os.path.join(folder, ".env")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
            return


_load_dotenv()

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Which strategy to use:
#   "remote" -> every task to Fireworks (a guaranteed-valid submission)
#   "router" -> try the free local model first, escalate only if unsure
# START with "remote" so you have a working submission, then switch to "router"
# once the real local model tier is added and tested. Override with AGENT_MODE.
MODE = os.environ.get("AGENT_MODE", "remote")


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    ledger = TokenLedger()
    results = []
    for task in tasks:
        # Map the harness's field names onto what our agent expects internally.
        internal_task = {"id": task["task_id"], "prompt": task["prompt"]}
        try:
            answer = run_agent(internal_task, ledger, mode=MODE)
        except Exception as exc:
            # Never crash the whole run over one task — record something valid.
            answer = f"ERROR: {exc}"
        results.append({"task_id": task["task_id"], "answer": answer})

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f)

    # For our own insight (the judging proxy counts the official tokens, not us).
    print(f"Done. {len(results)} answers written to {OUTPUT_PATH}.", file=sys.stderr)
    print(ledger.summary(), file=sys.stderr)


if __name__ == "__main__":
    main()
