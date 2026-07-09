"""
Token counting and accounting.

The ONLY thing that costs us on the leaderboard is REMOTE tokens.
Local tokens are free. So this ledger tracks remote and local usage
separately, and `remote_total` is the number we work to drive DOWN.

NOTE: count_tokens() below is a rough estimate — good enough for building
and testing the pipeline. At kickoff, swap it for the real tokenizer, or
better, just use the token counts the Fireworks API returns in its response.
"""

from dataclasses import dataclass, field


def count_tokens(text: str) -> int:
    """Rough token estimate: about 1 token per 4 characters.

    Real tokenizers differ, but this is fine for wiring things up.
    Replace with the real count at kickoff.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class TokenLedger:
    """Keeps a running total of tokens used, split by remote vs local."""

    remote_total: int = 0                      # <-- this is what we minimize
    local_total: int = 0                       # free; tracked only for our insight
    calls: list = field(default_factory=list)  # a log of every model call

    def add_remote(self, prompt_tokens: int, completion_tokens: int, note: str = ""):
        total = prompt_tokens + completion_tokens
        self.remote_total += total
        self.calls.append({
            "where": "remote", "prompt": prompt_tokens,
            "completion": completion_tokens, "total": total, "note": note,
        })

    def add_local(self, prompt_tokens: int, completion_tokens: int, note: str = ""):
        total = prompt_tokens + completion_tokens
        self.local_total += total
        self.calls.append({
            "where": "local", "prompt": prompt_tokens,
            "completion": completion_tokens, "total": total, "note": note,
        })

    def summary(self) -> str:
        return (
            f"remote tokens (the cost): {self.remote_total}   |   "
            f"local tokens (free): {self.local_total}   |   "
            f"total model calls: {len(self.calls)}"
        )
