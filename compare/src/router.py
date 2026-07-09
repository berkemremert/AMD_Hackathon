"""
The router — the brain that decides local vs remote.

Strategy: try the FREE local model first, estimate how confident it is, and
only escalate to the PAID remote model when the local answer looks shaky.

Confidence via self-consistency (free, because it's all local):
  run the local model several times on the same task and see how much the
  answers agree. Strong agreement -> confident. Scattered answers -> unsure.

The threshold is the single most important dial in the whole project:
  higher threshold  -> escalate more often -> more accurate, more tokens
  lower threshold   -> escalate less often -> fewer tokens, risk losing accuracy
Part 4 will sweep this threshold to find the sweet spot.
"""

from collections import Counter

from models import local_solve, remote_solve


def local_answer_with_confidence(task, samples=5):
    """Run the local model `samples` times; return (best_answer, confidence, usages).

    confidence = (votes for the most common answer) / samples, a number from 0 to 1.
    This is all FREE — it only uses the local model.
    """
    answers = []
    usages = []
    for _ in range(samples):
        answer, usage = local_solve(task)
        answers.append(answer)
        usages.append(usage)

    counts = Counter(answers)
    best_answer, votes = counts.most_common(1)[0]
    confidence = votes / samples
    return best_answer, confidence, usages


def route(task, ledger, threshold=0.6, samples=5):
    """Decide local vs remote for one task. Returns the final answer.

    1. Ask the local model several times and measure confidence — free.
    2. Confident enough (>= threshold)? Keep the local answer, 0 remote tokens.
    3. Otherwise escalate to the remote model and pay for those tokens.
    """
    local_ans, confidence, usages = local_answer_with_confidence(task, samples)

    # Record the local samples — free, tracked only for our own insight.
    for u in usages:
        ledger.add_local(u["prompt_tokens"], u["completion_tokens"], note=task["id"])

    if confidence >= threshold:
        return local_ans   # trusted the free model, spent 0 remote tokens

    # Not confident -> escalate to the paid remote model.
    remote_ans, usage = remote_solve(task)
    ledger.add_remote(usage["prompt_tokens"], usage["completion_tokens"], note=task["id"])
    return remote_ans
