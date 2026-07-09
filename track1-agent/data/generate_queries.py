"""Generates a labeled-ground-truth query set across the hackathon's 8 official
capability categories. Each query ships with an explicit ground_truth/rubric so
label_dataset.py can grade answers against a known target instead of pure vibes.

Every category has an "easy" and "hard" template pool. The pool label is only a
prior — the real easy/hard label used for training comes from label_dataset.py,
which measures whether the cheap model actually got it right.
"""
import json
import random
from pathlib import Path

random.seed(7)

OUT_PATH = Path(__file__).parent / "queries_raw.json"

NAMES = ["Priya", "Marcus", "Elena", "Kofi", "Yuki", "Sana", "Diego", "Fatima", "Liam", "Noor"]
CITIES = ["Nairobi", "Lisbon", "Austin", "Seoul", "Toronto", "Accra", "Warsaw", "Manila"]
COMPANIES = ["Northwind Traders", "Vantage Robotics", "Solace Health", "Brightline Logistics"]


def _q(category, difficulty_pool, prompt, ground_truth):
    return {"category": category, "difficulty_pool": difficulty_pool, "prompt": prompt, "ground_truth": ground_truth}


def gen_factual_knowledge():
    facts = [
        ("photosynthesis", "the process plants use to convert light energy into chemical energy, producing oxygen"),
        ("DNS", "the system that translates human-readable domain names into IP addresses"),
        ("compound interest", "interest calculated on both the principal and previously earned interest"),
        ("a REST API", "an interface that uses HTTP methods to let clients read and modify server resources statelessly"),
        ("inflation", "a sustained rise in the general price level that reduces purchasing power over time"),
        ("a black hole", "a region of spacetime where gravity is so strong nothing, not even light, can escape"),
    ]
    out = []
    for topic, key_idea in facts:
        out.append(_q("factual_knowledge", "easy", f"What is {topic}? Explain in one or two sentences.", key_idea))
        out.append(_q(
            "factual_knowledge", "hard",
            f"Explain {topic} in under 40 words, then give one concrete real-world example of it in action.",
            key_idea,
        ))
    return out


def gen_math_reasoning():
    out = []
    for _ in range(6):
        a, b = random.randint(20, 900), random.randint(20, 900)
        out.append(_q("math_reasoning", "easy", f"What is {a} + {b}?", str(a + b)))
    for _ in range(6):
        price = random.randint(40, 300)
        discount = random.choice([10, 15, 20, 25, 30])
        tax = random.choice([5, 7, 8])
        discounted = price * (1 - discount / 100)
        final = round(discounted * (1 + tax / 100), 2)
        out.append(_q(
            "math_reasoning", "hard",
            f"An item costs ${price}. It has a {discount}% discount applied, then {tax}% sales tax is "
            f"added to the discounted price. What is the final price, rounded to 2 decimal places?",
            str(final),
        ))
    return out


def gen_sentiment():
    positive = [
        "The onboarding flow was smooth and the support team replied within minutes.",
        "This laptop's battery easily lasts a full workday, and the screen is gorgeous.",
    ]
    negative = [
        "The app crashed three times during checkout and support never responded.",
        "The hotel room smelled of smoke and the AC was broken all weekend.",
    ]
    mixed_sarcastic = [
        "Oh sure, a 'premium' subscription that still shows ads every two minutes, love that.",
        "Great, another 'quick update' that took the app down for six hours. Really convenient.",
    ]
    out = []
    for t in positive:
        out.append(_q("sentiment_classification", "easy", f"Classify the sentiment of this review and justify it in one sentence: \"{t}\"", "positive"))
    for t in negative:
        out.append(_q("sentiment_classification", "easy", f"Classify the sentiment of this review and justify it in one sentence: \"{t}\"", "negative"))
    for t in mixed_sarcastic:
        out.append(_q(
            "sentiment_classification", "hard",
            f"Classify the sentiment of this review and justify it in one sentence, watching for sarcasm: \"{t}\"",
            "negative",
        ))
    return out


def gen_summarization():
    easy_texts = [
        (
            "The city council approved a new bike lane network yesterday, covering 12 miles of "
            "downtown streets. Construction begins next spring and is expected to finish by fall.",
            "12-mile bike lane network approved, construction spring to fall",
        ),
        (
            "A local bakery chain announced it will open five new locations across the metro area "
            "this year, creating roughly 60 new jobs and focusing on gluten-free product lines.",
            "bakery chain opening 5 new locations, ~60 jobs, gluten-free focus",
        ),
        (
            "Researchers found that a new coating reduces bacterial growth on hospital door handles "
            "by 90%, and the material costs about the same as standard paint to apply.",
            "new coating cuts bacterial growth on door handles by 90%, similar cost to standard paint",
        ),
    ]
    hard_texts = [
        (
            "Quarterly revenue rose 8% year over year, driven mainly by the enterprise segment, which grew "
            "14% as three large customers renewed multi-year contracts. The consumer segment was roughly flat, "
            "with a 1% decline attributed to seasonal churn in the mobile app. Operating costs increased 5%, "
            "primarily from headcount growth in customer success. Management reiterated full-year guidance and "
            "flagged supply chain costs as the main risk for the next two quarters.",
            "revenue up 8% on enterprise growth; costs up 5% on headcount, supply chain risk flagged",
        ),
        (
            "The city's transit authority reported a 22% rise in ridership after launching a flat monthly "
            "pass, though fare revenue per trip fell 9% as riders shifted from single tickets. Maintenance "
            "costs also rose due to an aging bus fleet, prompting the authority to request emergency funding "
            "from the state to cover a projected budget gap next fiscal year.",
            "ridership up 22% on flat pass; fare revenue per trip down 9%, maintenance costs up, funding gap looming",
        ),
        (
            "A two-year study of remote workers found that productivity, measured by completed tasks, held "
            "steady compared to in-office baselines, but self-reported feelings of isolation rose sharply "
            "among employees who joined the company entirely remotely. Companies that ran regular in-person "
            "meetups saw smaller increases in reported isolation than those that stayed fully virtual.",
            "remote productivity held steady; isolation rose especially for remote-only hires, mitigated by in-person meetups",
        ),
    ]
    out = []
    for text, gt in easy_texts:
        out.append(_q("text_summarization", "easy", f"Summarize this in one sentence: {text}", gt))
    for text, gt in hard_texts:
        out.append(_q(
            "text_summarization", "hard",
            f"Summarize this in exactly 2 bullet points, each under 15 words: {text}",
            gt,
        ))
    return out


def gen_ner():
    out = []
    for _ in range(4):
        name, city, company = random.choice(NAMES), random.choice(CITIES), random.choice(COMPANIES)
        date = f"{random.choice(['March','June','September','November'])} {random.randint(1,28)}, 2025"
        sentence = f"{name} met with representatives from {company} in {city} on {date} to finalize the merger."
        out.append(_q(
            "named_entity_recognition", "easy",
            f"Extract all named entities (person, organization, location, date) from this sentence: \"{sentence}\"",
            f"person={name}; org={company}; location={city}; date={date}",
        ))
    for _ in range(4):
        n1, n2 = random.sample(NAMES, 2)
        c1, c2 = random.sample(COMPANIES, 2)
        loc1, loc2 = random.sample(CITIES, 2)
        sentence = (
            f"{n1} of {c1} and {n2} of {c2} announced a joint venture headquartered in {loc1}, "
            f"with a secondary office opening in {loc2} next quarter."
        )
        out.append(_q(
            "named_entity_recognition", "hard",
            f"Extract all named entities (person, organization, location) from this sentence, "
            f"labeling each by type: \"{sentence}\"",
            f"person={n1},{n2}; org={c1},{c2}; location={loc1},{loc2}",
        ))
    return out


def gen_code_debugging():
    easy_snippets = [
        (
            "def total(nums):\n    result = 0\n    for i in range(1, len(nums)):\n        result += nums[i]\n    return result",
            "off-by-one: loop starts at index 1 instead of 0, so it skips nums[0]",
        ),
        (
            "def is_even(n):\n    if n % 2 == 1:\n        return True\n    return False",
            "inverted condition: returns True for odd numbers instead of even",
        ),
        (
            "def find_max(nums):\n    max_val = 0\n    for n in nums:\n        if n > max_val:\n            max_val = n\n    return max_val",
            "wrong initial value: seeding max_val at 0 fails for lists where every number is negative",
        ),
        (
            "def get_first(items):\n    return items[0]\n\ndef safe_first(items):\n    return get_first(items) if items != None else None",
            "checks `items != None` but not for an empty list, so get_first still raises IndexError on []",
        ),
    ]
    hard_snippets = [
        (
            "def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket",
            "mutable default argument: the list is shared and grows across calls instead of resetting",
        ),
        (
            "def average(nums):\n    return sum(nums) / len(nums) if nums else 0\n\ndef report(groups):\n"
            "    return [average(g) for g in groups if g == []]",
            "the filter condition `if g == []` excludes every non-empty group, so report() always returns an empty list",
        ),
        (
            "def dedupe_preserve_order(items):\n    seen = set()\n    result = []\n    for item in items:\n"
            "        if item not in seen:\n            result.append(item)\n        seen.add(item)\n    return result",
            "logic order bug: item is added to `seen` unconditionally, but since it's added after the check, "
            "this actually works; the real bug is it silently fails on unhashable items like lists, no error handling",
        ),
        (
            "class Counter:\n    counts = {}\n    def add(self, key):\n        self.counts[key] = self.counts.get(key, 0) + 1\n"
            "    def get(self, key):\n        return self.counts.get(key, 0)",
            "class-level mutable attribute: `counts` is shared across all Counter instances instead of being per-instance",
        ),
    ]
    out = []
    for code, bug in easy_snippets:
        out.append(_q("code_debugging", "easy", f"Find and explain the bug in this Python function:\n```python\n{code}\n```", bug))
    for code, bug in hard_snippets:
        out.append(_q("code_debugging", "hard", f"Find and explain the bug in this Python code, and propose a fix:\n```python\n{code}\n```", bug))
    return out


def gen_logic_puzzle():
    out = []
    out.append(_q(
        "logical_reasoning", "easy",
        "Three friends, Ana, Ben, and Cleo, each own a different pet: a cat, a dog, or a fish. "
        "Ana does not own the dog. Ben does not own the fish. Cleo owns the cat. "
        "Who owns the dog?",
        "Ben",
    ))
    out.append(_q(
        "logical_reasoning", "easy",
        "Four coworkers sit in a row: Dara, Eli, Fay, and Gus. Dara sits immediately left of Eli. "
        "Gus sits at the far right. Fay sits at the far left. Who sits second from the left?",
        "Dara",
    ))
    out.append(_q(
        "logical_reasoning", "easy",
        "Three students, Omar, Priya, and Quinn, each scored differently on a test: 70, 85, or 92. "
        "Omar did not score the highest. Quinn scored higher than Omar. Priya scored the highest. "
        "What did Omar score?",
        "70",
    ))
    out.append(_q(
        "logical_reasoning", "easy",
        "Two boxes are labeled A and B. Box A is heavier than box B. Box C is lighter than box B. "
        "Which box is the lightest?",
        "Box C",
    ))
    out.append(_q(
        "logical_reasoning", "hard",
        "Five houses in a row are painted different colors: red, blue, green, yellow, white. "
        "The green house is immediately left of the white house. The red house is at one end. "
        "The blue house is exactly in the middle. The yellow house is immediately right of the blue house. "
        "The green house is not adjacent to the red house. What color is the house at the far right?",
        "red",
    ))
    out.append(_q(
        "logical_reasoning", "hard",
        "Four runners, Tia, Uma, Vik, and Wes, finished a race with no ties. Tia finished before Uma. "
        "Wes finished immediately after Vik. Vik finished first. "
        "What is the finishing order from first to last?",
        "Vik, Wes, Tia, Uma",
    ))
    out.append(_q(
        "logical_reasoning", "hard",
        "Five colleagues, Alex, Bo, Cid, Dee, and Fen, each work in a different department: "
        "Sales, HR, Legal, IT, or Finance. Alex works in neither Sales nor IT. "
        "Bo works in Legal. Cid works in Finance. "
        "Dee does not work in HR or Sales. Fen works in Sales. "
        "Given these constraints, which department does Alex work in?",
        "HR",
    ))
    out.append(_q(
        "logical_reasoning", "hard",
        "Four boxes, W, X, Y, Z, contain different numbers of marbles: 3, 6, 9, and 12, in some order. "
        "Box W has more marbles than box X. Box X has more marbles than box Y. "
        "Box Z has more marbles than all the others. "
        "How many marbles are in box X?",
        "6",
    ))
    return out


def gen_code_generation():
    out = []
    out.append(_q(
        "code_generation", "easy",
        "Write a Python function `is_palindrome(s)` that returns True if the string s reads the same "
        "forwards and backwards (ignore case), else False.",
        json.dumps({
            "function_name": "is_palindrome",
            "tests": [
                {"args": ["racecar"], "expected": True},
                {"args": ["Hello"], "expected": False},
                {"args": ["Level"], "expected": True},
            ],
        }),
    ))
    out.append(_q(
        "code_generation", "easy",
        "Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, "
        "case-insensitive) in the string s.",
        json.dumps({
            "function_name": "count_vowels",
            "tests": [
                {"args": ["Hello World"], "expected": 3},
                {"args": ["xyz"], "expected": 0},
            ],
        }),
    ))
    out.append(_q(
        "code_generation", "hard",
        "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] integer "
        "pairs and returns a list of merged, non-overlapping intervals sorted by start.",
        json.dumps({
            "function_name": "merge_intervals",
            "tests": [
                {"args": [[[1, 3], [2, 6], [8, 10], [15, 18]]], "expected": [[1, 6], [8, 10], [15, 18]]},
                {"args": [[[1, 4], [4, 5]]], "expected": [[1, 5]]},
            ],
        }),
    ))
    out.append(_q(
        "code_generation", "hard",
        "Write a Python function `group_anagrams(words)` that groups a list of strings into lists of "
        "anagrams of each other. Return a list of groups, each group sorted alphabetically, and the "
        "groups sorted by their first element.",
        json.dumps({
            "function_name": "group_anagrams",
            "tests": [
                {"args": [["eat", "tea", "tan", "ate", "nat", "bat"]],
                 "expected": [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]},
            ],
        }),
    ))
    return out


def main():
    generators = [
        gen_factual_knowledge, gen_math_reasoning, gen_sentiment, gen_summarization,
        gen_ner, gen_code_debugging, gen_logic_puzzle, gen_code_generation,
    ]
    queries = []
    for gen in generators:
        queries.extend(gen())
    for i, q in enumerate(queries):
        q["id"] = f"q{i:03d}"
    random.shuffle(queries)
    OUT_PATH.write_text(json.dumps(queries, indent=2))
    print(f"Wrote {len(queries)} queries to {OUT_PATH}")
    by_cat = {}
    for q in queries:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
