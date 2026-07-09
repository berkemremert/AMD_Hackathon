"""
The agent = our decision pipeline.

Modes:
  - "remote": send every task to the paid remote model (baseline)
  - "local":  send every task to the free local model  (baseline)
  - "router": try local first, check confidence, escalate to remote only if unsure
              -> this is our real strategy (see router.py)

The router uses two settings:
  threshold — how confident the local model must be to skip the remote model
  samples   — how many times to ask the local model when measuring confidence
"""

from models import remote_solve, local_solve
from router import route


def run_agent(task, ledger, mode="router", threshold=0.6, samples=5):
    """Solve one task in the given mode, recording token usage. Returns the answer."""
    if mode == "remote":
        answer, usage = remote_solve(task)
        # Remote tokens are the ones that cost us on the leaderboard.
        ledger.add_remote(usage["prompt_tokens"], usage["completion_tokens"], note=task["id"])
        return answer

    if mode == "local":
        answer, usage = local_solve(task)
        # Local tokens are free — recorded only for our own insight.
        ledger.add_local(usage["prompt_tokens"], usage["completion_tokens"], note=task["id"])
        return answer

    if mode == "router":
        return route(task, ledger, threshold=threshold, samples=samples)

    raise ValueError(f"Unknown mode: {mode!r}. Use 'remote', 'local', or 'router'.")
