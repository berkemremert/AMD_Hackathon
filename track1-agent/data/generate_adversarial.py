"""Top-up batch of genuinely adversarial queries.

The first labeling pass showed minimax-m3 (our cheap tier) passing almost
everything from generate_queries.py's "hard" pool, leaving only 2 hard labels
out of 58 - not enough to train a router on. This script adds queries designed
to actually be hard for a capable MoE model: multi-step math, dense logic
puzzles, subtle Python bugs, algorithmic code generation, ambiguous NER/
sentiment, and strict multi-constraint summarization. Every ground_truth here
was independently verified (brute force for puzzles, reference implementations
for code) before being written down - see the session's verification steps.
"""
import json
from pathlib import Path

QUERIES_PATH = Path(__file__).parent / "queries_raw.json"


def _q(category, prompt, ground_truth):
    return {"category": category, "difficulty_pool": "hard", "prompt": prompt, "ground_truth": ground_truth}


def build():
    out = []

    # --- math_reasoning: multi-step, order-sensitive ---
    out.append(_q(
        "math_reasoning",
        "A tank starts with 480 liters. It drains at 8 liters per minute for 15 minutes, then is "
        "refilled at 12 liters per minute for 20 minutes, then drains again at 5 liters per minute "
        "for 10 minutes. How many liters are in the tank now?",
        "550",
    ))
    out.append(_q(
        "math_reasoning",
        "Pipe A can fill a pool in 6 hours. Pipe B can fill the same pool in 4 hours. If both pipes "
        "are opened together, how many hours will it take to fill the pool? Answer as a decimal "
        "rounded to 1 decimal place.",
        "2.4",
    ))
    out.append(_q(
        "math_reasoning",
        "A price is increased by 20%, then decreased by 20%, then increased by 10%. If the original "
        "price was $200, what is the final price?",
        "$211.20",
    ))
    out.append(_q(
        "math_reasoning",
        "You invest $1000. In year 1 it grows by 5%, in year 2 by 8%, in year 3 it shrinks by 3%. "
        "What is the final amount, rounded to 2 decimal places?",
        "$1099.98",
    ))

    # --- logical_reasoning: dense multi-constraint puzzles, brute-force verified unique ---
    out.append(_q(
        "logical_reasoning",
        "Five friends, Ivy, Jude, Kai, Lena, and Moss, sit in five seats numbered 1 to 5 in a row. "
        "Ivy sits in seat 1. Moss sits in seat 5. The person who drinks tea sits immediately to the "
        "right of Ivy. Kai drinks water and sits immediately to the left of Lena. What is the "
        "seating order from seat 1 to seat 5?",
        "Ivy, Jude, Kai, Lena, Moss",
    ))
    out.append(_q(
        "logical_reasoning",
        "Five boxes, P, Q, R, S, T, contain weights of 2, 4, 6, 8, and 10 kg, one weight per box. "
        "Box P and box Q together weigh 6 kg. Box R is heavier than box S. Box T is the heaviest "
        "of all five boxes. Box S and box Q differ in weight by exactly 2 kg. What is the weight "
        "of box R, in kg?",
        "8",
    ))

    # --- code_debugging: subtle Python semantics, not surface-level typos ---
    out.append(_q(
        "code_debugging",
        "What does `results` evaluate to, and why, given this code?\n```python\n"
        "def make_multipliers():\n    return [lambda x: x * i for i in range(1, 4)]\n\n"
        "multipliers = make_multipliers()\nresults = [m(10) for m in multipliers]\n```",
        "results = [30, 30, 30], not [10, 20, 30], because all three lambdas close over the same "
        "variable i by reference (late binding), and i equals 3 for all of them once the loop finishes",
    ))
    out.append(_q(
        "code_debugging",
        "What do `total1` and `total2` evaluate to, and why, given this code?\n```python\n"
        "def get_evens(nums):\n    return (n for n in nums if n % 2 == 0)\n\n"
        "evens = get_evens([1, 2, 3, 4, 5, 6])\ntotal1 = sum(evens)\ntotal2 = sum(evens)\n```",
        "total1 = 12 and total2 = 0, because get_evens returns a generator, and generators are "
        "exhausted after being consumed once, so the second sum() call has nothing left to iterate",
    ))
    out.append(_q(
        "code_debugging",
        "Given `prices = [0.1, 0.2]`, what does `check_total(prices)` return, and why?\n```python\n"
        "def check_total(prices):\n    total = sum(prices)\n    return total == 0.3\n```",
        "returns False, because summing 0.1 and 0.2 in binary floating point gives "
        "0.30000000000000004, which is not exactly equal to the float 0.3",
    ))
    out.append(_q(
        "code_debugging",
        "What happens when you call `factorial(0)` with this code, and why?\n```python\n"
        "def factorial(n):\n    if n == 1:\n        return 1\n    return n * factorial(n - 1)\n```",
        "it causes infinite recursion (eventually a RecursionError), because the base case only "
        "checks n == 1, not n <= 1, so calling with n=0 recurses through negative numbers forever",
    ))

    # --- code_generation: real algorithms, graded by executing the code ---
    out.append(_q(
        "code_generation",
        "Write a Python function `edit_distance(a, b)` that returns the minimum number of single-"
        "character insertions, deletions, or substitutions required to transform string a into "
        "string b (Levenshtein distance).",
        json.dumps({
            "function_name": "edit_distance",
            "tests": [
                {"args": ["kitten", "sitting"], "expected": 3},
                {"args": ["", "abc"], "expected": 3},
                {"args": ["abc", "abc"], "expected": 0},
            ],
        }),
    ))
    out.append(_q(
        "code_generation",
        "Write a Python function `longest_increasing_subsequence_length(nums)` that returns the "
        "length of the longest strictly increasing subsequence in a list of integers.",
        json.dumps({
            "function_name": "longest_increasing_subsequence_length",
            "tests": [
                {"args": [[10, 9, 2, 5, 3, 7, 101, 18]], "expected": 4},
                {"args": [[]], "expected": 0},
                {"args": [[5, 4, 3, 2, 1]], "expected": 1},
            ],
        }),
    ))

    # --- named_entity_recognition: deliberately ambiguous names ---
    out.append(_q(
        "named_entity_recognition",
        "Extract all named entities from this sentence, labeling each by type (person, organization, "
        "location, date), noting that some words could plausibly be more than one type: "
        "\"Amazon announced that Jordan, the new VP hired from Washington, will lead the Phoenix "
        "office starting in April.\"",
        "person=Jordan; organization=Amazon; location=Washington, Phoenix; date=April "
        "(Jordan, Washington, and Phoenix are all also place/country names, but here Jordan is a "
        "person and Washington/Phoenix are locations)",
    ))
    out.append(_q(
        "named_entity_recognition",
        "Extract all named entities from this sentence, labeling each by type: \"Turner joined "
        "Sterling Bank in Sterling, Colorado, replacing Bell who moved to a Bell Labs research "
        "role in June.\"",
        "person=Turner, Bell; organization=Sterling Bank, Bell Labs; location=Sterling, Colorado; "
        "date=June (Sterling and Bell each appear twice as different entity types in the same sentence)",
    ))

    # --- sentiment_classification: backhanded / double-negative phrasing ---
    out.append(_q(
        "sentiment_classification",
        "Classify the sentiment of this review and justify it in one sentence: \"I wouldn't say "
        "the food was bad, exactly, but I also wouldn't rush back.\"",
        "negative (a double-negative hedge that still lands as lukewarm-to-negative; the reviewer "
        "is avoiding rushing back, a clear signal of dissatisfaction despite the softened wording)",
    ))
    out.append(_q(
        "sentiment_classification",
        "Classify the sentiment of this review and justify it in one sentence: \"Well, at least it "
        "wasn't the WORST customer service I've ever had.\"",
        "negative (sarcastic backhanded phrasing; damning with faint praise still signals a poor experience)",
    ))

    # --- text_summarization: strict multi-constraint format compliance ---
    out.append(_q(
        "text_summarization",
        "Summarize this in exactly 3 sentences, each under 12 words, making sure both $4.2 million "
        "and 50,000 users appear somewhere in the summary: \"The startup raised $4.2 million in "
        "seed funding and grew its user base to 50,000 monthly active users within six months, "
        "driven mainly by a viral referral program that cut acquisition costs by half.\"",
        "exactly 3 sentences, each under 12 words, must mention both $4.2 million and 50,000 users",
    ))

    # --- factual_knowledge: precise technical distinctions, easy to conflate ---
    out.append(_q(
        "factual_knowledge",
        "Explain the difference between concurrency and parallelism in under 35 words, then give "
        "one example of a program that is concurrent but not parallel.",
        "concurrency is managing multiple tasks with overlapping progress, not necessarily at the "
        "exact same instant; parallelism is executing multiple tasks literally simultaneously on "
        "multiple cores; example: a single-core async event loop interleaving I/O-bound tasks",
    ))
    out.append(_q(
        "factual_knowledge",
        "Explain the difference between a process and a thread in under 35 words, focused on memory.",
        "each process has its own isolated memory space; threads within the same process share that "
        "process's memory space, so they can directly read and write the same variables",
    ))

    return out


def main():
    existing = json.loads(QUERIES_PATH.read_text())
    max_num = max(int(q["id"][1:]) for q in existing)
    new_queries = build()
    for i, q in enumerate(new_queries):
        q["id"] = f"q{max_num + 1 + i:03d}"
    existing.extend(new_queries)
    QUERIES_PATH.write_text(json.dumps(existing, indent=2))
    print(f"Added {len(new_queries)} adversarial queries (total now {len(existing)})")


if __name__ == "__main__":
    main()
