"""Interactive prompt loop for FrugalRouter.

Launch once, then type prompts freely:

    python -m frugalrouter.repl
    python -m frugalrouter.repl --model accounts/fireworks/models/kimi-k2p7-code
    python -m frugalrouter.repl --local          # local shortcuts only, zero tokens

In-session commands: :model <ref>, :local on|off, :help, :quit
"""

from __future__ import annotations

import argparse
import os

from frugalrouter.config import load_config
from frugalrouter.router import FrugalRouter
from frugalrouter.types import Task


BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"

HELP = """Commands:
  :model <ref>    switch model (e.g. a deployment ref for a Gemma model)
  :model          show the current model
  :local on       answer with local shortcuts only (zero tokens)
  :local off      re-enable remote model calls
  :help           show this help
  :quit           exit (Ctrl-C also works)
Anything else is sent to the router as a prompt."""


def build_router(model: str, allow_remote: bool) -> FrugalRouter:
    os.environ.setdefault("FIREWORKS_BASE_URL", BASE_URL)
    os.environ["ALLOWED_MODELS"] = model
    config = load_config("config/models.json")
    return FrugalRouter(config=config, allow_remote=allow_remote)


def run_one(router: FrugalRouter, prompt: str, counter: int) -> None:
    result = router.run(Task(id=f"repl-{counter}", input=prompt))
    answer = result.answer
    print(answer.text if answer.text else "(empty answer)")
    if result.used_remote:
        total = answer.prompt_tokens + answer.completion_tokens
        meta = (
            f"{result.route} | {result.category} | {answer.model} | "
            f"{total} tokens ({answer.prompt_tokens}+{answer.completion_tokens}) | {answer.latency_ms}ms"
        )
    else:
        meta = f"{result.route} | {result.category} | 0 tokens | {answer.latency_ms}ms"
    print(f"  -> {meta}")
    if result.fallback_reason and "remote_error" in result.fallback_reason:
        print(f"  ! {result.fallback_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive FrugalRouter prompt loop.")
    parser.add_argument("--model", default=os.environ.get("ALLOWED_MODELS", DEFAULT_MODEL))
    parser.add_argument("--local", action="store_true", help="Local shortcuts only; no remote calls.")
    args = parser.parse_args()

    model = args.model.split(",")[0].strip()
    allow_remote = not args.local
    router = build_router(model, allow_remote)

    if allow_remote and not os.environ.get("FIREWORKS_API_KEY"):
        print("Warning: FIREWORKS_API_KEY is not set — remote calls will fail (local shortcuts still work).\n")

    print(f"FrugalRouter REPL - model: {model}{' [local only]' if args.local else ''}")
    print("Type a prompt and press Enter. :help for commands, Ctrl-C to quit.\n")

    counter = 0
    while True:
        try:
            line = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not line:
            continue
        if line in {":quit", ":exit", ":q"}:
            print("bye")
            return
        if line == ":help":
            print(HELP)
            continue
        if line.startswith(":model"):
            rest = line[len(":model"):].strip()
            if not rest:
                print(f"current model: {model}")
            else:
                model = rest
                router = build_router(model, allow_remote)
                print(f"switched to: {model}")
            continue
        if line.startswith(":local"):
            arg = line[len(":local"):].strip().lower()
            if arg == "on":
                allow_remote = False
            elif arg == "off":
                allow_remote = True
            router = build_router(model, allow_remote)
            print(f"local-only: {not allow_remote}")
            continue

        counter += 1
        try:
            run_one(router, line, counter)
        except Exception as error:  # noqa: BLE001 - surface any error, keep the loop alive
            print(f"  ! error: {error}")
        print()


if __name__ == "__main__":
    main()
