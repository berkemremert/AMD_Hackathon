from __future__ import annotations

from datetime import date
import re
import time
from dataclasses import dataclass

from frugalrouter.math_solver import solve_simple_math
from frugalrouter.types import Answer, Task


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

US_STATE_NAMES = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
]

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass(frozen=True)
class LocalCandidate:
    answer: Answer
    confidence: float
    reasons: list[str]


class LocalProvider:
    name = "local"
    model = "deterministic-shortcuts"

    def answer(self, task: Task) -> LocalCandidate:
        started = time.perf_counter()
        text, confidence, reasons = self._answer_text(task)
        latency_ms = int((time.perf_counter() - started) * 1000)
        answer = Answer(text=text, provider=self.name, model=self.model, latency_ms=latency_ms)
        return LocalCandidate(answer=answer, confidence=confidence, reasons=reasons)

    def _answer_text(self, task: Task) -> tuple[str, float, list[str]]:
        prompt = task.input.strip()
        lower = prompt.lower()

        exact = self._exact_response(prompt)
        if exact is not None:
            return exact, 0.99, ["exact_response_instruction"]

        if self._is_sentiment_prompt(lower):
            return self._sentiment(prompt)

        code_debugging = self._code_debugging_shortcut(prompt)
        if code_debugging is not None:
            return code_debugging, 0.99, ["computed_loop_update"]

        factual = self._factual_shortcut(prompt)
        if factual is not None:
            return factual, 0.99, ["computed_factual_shortcut"]

        logic = self._logic_shortcut(prompt)
        if logic is not None:
            return logic, 0.99, ["computed_logic_shortcut"]

        calendar = self._calendar_weekday(prompt)
        if calendar is not None:
            return calendar, 0.99, ["computed_calendar_weekday"]

        time_answer = self._time_word_problem(prompt)
        if time_answer is not None:
            return time_answer, 0.98, ["computed_time_word_problem"]

        math_answer = solve_simple_math(prompt)
        if math_answer is not None:
            text, reason = math_answer
            return text, 0.97, [reason]

        return "", 0.0, ["no_local_shortcut"]

    NEGATION_MARKERS = {"not", "never", "no", "hardly", "barely", "isn't", "wasn't", "aren't", "won't", "don't", "doesn't", "didn't", "can't", "couldn't", "nothing", "neither", "nor", "lacks", "without"}

    def _sentiment(self, prompt: str) -> tuple[str, float, list[str]]:
        positive = {"good", "great", "fast", "clear", "useful", "love", "excellent", "happy", "impressed"}
        negative = {"bad", "slow", "confusing", "broken", "hate", "poor", "wrong", "sad", "terrible", "awful"}
        words = set(re.findall(r"[a-zA-Z']+", prompt.lower()))
        if words & self.NEGATION_MARKERS:
            return "", 0.0, ["sentiment_negation_present"]
        # One keyword is too weak (sarcasm: "Oh great, another crash");
        # require a clear multi-keyword majority before answering locally.
        score = len(words & positive) - len(words & negative)
        if score >= 2:
            return "positive", 0.94, ["clear_sentiment_keywords"]
        if score <= -2:
            return "negative", 0.94, ["clear_sentiment_keywords"]
        return "", 0.0, ["no_clear_sentiment_keywords"]

    def _is_sentiment_prompt(self, prompt: str) -> bool:
        return "sentiment" in prompt and any(word in prompt for word in ["classify", "label", "positive", "negative"])

    def _exact_response(self, prompt: str) -> str | None:
        match = re.search(
            r"(?:reply|respond|answer)\s+with\s+exactly\s*(?::\s*([^\n.]+)|['\"]([^'\"]+)['\"])",
            prompt,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        # "Answer with exactly 'yes' or 'no': ..." offers alternatives; the right
        # choice depends on the actual question, so it is not a safe shortcut.
        tail = prompt[match.end() :]
        if re.match(r"\s*(?:,|or\b|and\b)", tail, flags=re.IGNORECASE):
            return None
        literal = (match.group(1) or match.group(2)).strip()
        if re.search(r"\bor\b", literal, flags=re.IGNORECASE):
            return None
        return literal

    def _factual_shortcut(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if (
            "only letter" in lower
            and "not appear" in lower
            and ("u.s. state" in lower or "us state" in lower or "united states state" in lower)
        ):
            used = {char for state in US_STATE_NAMES for char in state.lower() if "a" <= char <= "z"}
            missing = [chr(code) for code in range(ord("a"), ord("z") + 1) if chr(code) not in used]
            if len(missing) == 1:
                return missing[0].upper()
        return None

    def _code_debugging_shortcut(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\b(?:missing|omit(?:ted)?|forgot(?:ten)?)\b", lower):
            return None
        if not re.search(r"\b(?:operation|update|increment|decrement)\b", lower):
            return None
        if not re.search(r"\b(?:freeze|freezes|frozen|infinite\s+loop)\b", lower):
            return None

        loop = re.search(
            r"\bwhile\s*\(\s*([A-Za-z_]\w*)\s*(<=?|>=?)\s*[^)]+\)\s*\{([^}]*)\}",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not loop:
            return None

        variable, comparison, body = loop.groups()
        if re.search(
            rf"\b{re.escape(variable)}\b\s*(?:\+\+|--|[+\-*/]?=)",
            body,
        ):
            return None
        return f"{variable}++" if comparison.startswith("<") else f"{variable}--"

    def _logic_shortcut(self, prompt: str) -> str | None:
        power_strip_answer = self._power_strip_empty_outlets(prompt)
        if power_strip_answer is not None:
            return power_strip_answer
        return None

    def _power_strip_empty_outlets(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not all(marker in lower for marker in ["power strip", "outlet"]):
            return None
        if "wall outlet" not in lower or not re.search(r"\bempty\b|\bavailable\b|\bremain", lower):
            return None

        strips = self._extract_number_before(r"(?:identical\s+)?power strips?", lower)
        outlets = self._extract_number_before(r"outlets?", lower)
        if strips is None or outlets is None or strips < 1 or outlets < 1:
            return None

        # First strip plugs into the wall; each additional strip consumes one
        # outlet on an earlier strip. Empty outlets = all strip outlets minus
        # the strip-to-strip connections.
        return str(strips * outlets - max(0, strips - 1))

    def _time_word_problem(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\b(?:what time|earliest time|completely done|finish(?:ed|es)?|ready)\b", lower):
            return None

        start = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b", lower, flags=re.IGNORECASE)
        if not start:
            return None

        duration_minutes = 0
        for match in re.finditer(
            r"\b(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+hours?)"
            r"(?:\s+and\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
            r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+minutes?)?"
            r"|\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
            r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+minutes?",
            lower,
        ):
            hours = self._parse_number(match.group(1)) if match.group(1) else 0
            minutes = self._parse_number(match.group(2) or match.group(3)) if (match.group(2) or match.group(3)) else 0
            duration_minutes += hours * 60 + minutes

        if duration_minutes <= 0:
            return None

        hour = int(start.group(1))
        minute = int(start.group(2) or 0)
        suffix = start.group(3).replace(".", "").lower()
        if hour == 12:
            hour = 0
        if suffix == "pm":
            hour += 12

        total = (hour * 60 + minute + duration_minutes) % (24 * 60)
        out_hour_24, out_minute = divmod(total, 60)
        out_suffix = "AM" if out_hour_24 < 12 else "PM"
        out_hour_12 = out_hour_24 % 12 or 12

        if re.search(r"\bHH:MM\b", prompt):
            return f"{out_hour_12:02d}:{out_minute:02d} {out_suffix}"
        return f"{out_hour_12}:{out_minute:02d} {out_suffix}"

    def _calendar_weekday(self, prompt: str) -> str | None:
        lower = prompt.lower()
        if not re.search(r"\bday of the week\b|\bwhat day\b", lower):
            return None

        date_pattern = (
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b"
        )
        matches = list(re.finditer(date_pattern, lower, flags=re.IGNORECASE))
        if len(matches) < 2:
            return None

        base_weekday = None
        base_context = lower[matches[0].end() : matches[0].end() + 80]
        for index, name in enumerate(WEEKDAYS):
            if re.search(rf"\b{name.lower()}\b", base_context):
                base_weekday = index
                break
        if base_weekday is None:
            return None

        base_date = self._date_from_match(matches[0])
        target_date = self._date_from_match(matches[-1])
        if base_date is None or target_date is None:
            return None

        offset = (target_date - base_date).days % 7
        return WEEKDAYS[(base_weekday + offset) % 7]

    def _extract_number_before(self, following_pattern: str, text: str) -> int | None:
        pattern = rf"\b(\d+|{'|'.join(NUMBER_WORDS)})\s+{following_pattern}"
        match = re.search(pattern, text)
        return self._parse_number(match.group(1)) if match else None

    def _parse_number(self, value: str) -> int:
        return int(value) if value.isdigit() else NUMBER_WORDS[value.lower()]

    def _date_from_match(self, match: re.Match[str]) -> date | None:
        try:
            return date(int(match.group(3)), MONTHS[match.group(1).lower()], int(match.group(2)))
        except ValueError:
            return None
