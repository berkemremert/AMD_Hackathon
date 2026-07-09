"""
Checking whether an answer is correct.

IMPORTANT: real Track 1 tasks may be graded differently (exact match,
a multiple-choice letter, "answer contains the expected text", etc.).
At kickoff, adjust is_correct() to match EXACTLY how the hackathon grades
answers — this matters, because it decides whether you're above the
accuracy line on the leaderboard.
"""


def normalize(text):
    return str(text).strip().lower()


def is_correct(answer, expected):
    """Loose check: is the expected answer found inside the model's answer?

    Works well for short-answer tasks. Swap for exact-match or MCQ-letter
    checking at kickoff if that's how Track 1 grades.
    """
    return normalize(expected) in normalize(answer)
