from frugalrouter.prompting import clean_answer, looks_like_reasoning_spill, user_prompt
from frugalrouter.types import Task


def test_detects_gemma_thought_spill():
    assert looks_like_reasoning_spill("thought\n*   The user wants X.")
    assert looks_like_reasoning_spill("We need answer user. They want a fix.")
    assert not looks_like_reasoning_spill("Canberra")
    assert not looks_like_reasoning_spill("def add_one(x):\n    return x + 1")


def test_no_reasoning_directive_appended():
    # factual is a non-reasoning category, where the directive applies.
    task = Task(id="t", input="What is the capital of France?")
    assert "Do not show reasoning" in user_prompt(task, "factual", no_reasoning=True)
    assert "Do not show reasoning" not in user_prompt(task, "factual")


def test_sentiment_prompt_warns_about_target_and_mixed_phrasing():
    task = Task(id="t", input="Evaluate the sentiment: I hated leaving this wonderful resort.")
    prompt = user_prompt(task, "sentiment")

    assert "asked-about target" in prompt
    assert "negation and mixed phrasing" in prompt


def test_ner_prompt_preserves_exact_source_spans():
    task = Task(id="t", input="Extract all dates: third of June, 2020.")
    prompt = user_prompt(task, "ner")

    assert "exact source spans" in prompt
    assert "do not normalize" in prompt


def test_reasoning_category_uses_brief_written_reasoning_by_default():
    # V13: one-shot silent answers lost hidden math/logic tasks (V12 = 14/19),
    # while unbounded hidden thinking cost ~1k tokens/task (V11 = 14,903 total).
    # Brief written steps ending in FINAL ANSWER recover accuracy cheaply.
    task = Task(id="t", input="A train travels 120 km in 2 hours; how long for 300 km?")
    for cat in ("math", "logic"):
        prompt = user_prompt(task, cat, no_reasoning=True)
        assert "Reason briefly" in prompt
        assert "FINAL ANSWER" in prompt


def test_reasoning_category_preserves_requested_explanation():
    task = Task(id="t", input="Show your work: what is 12 * 15?")
    for cat in ("math", "logic"):
        prompt = user_prompt(task, cat)
        assert "FINAL ANSWER" in prompt


def test_reasoning_category_extracts_final_answer():
    from frugalrouter.prompting import clean_answer

    reasoned = "Step 1: 120/2 = 60 km/h.\nStep 2: 300/60 = 5 h.\n**FINAL ANSWER:** 5 hours"
    assert clean_answer(reasoned, "math") == "5 hours"


def test_strips_single_code_fence():
    fenced = "```python\ndef add_one(x):\n    return x + 1\n```"
    assert clean_answer(fenced, "code_generation") == "def add_one(x):\n    return x + 1"


def test_strips_inline_code_ticks_for_short_exact_answers():
    assert clean_answer("`[]`", "general") == "[]"


def test_extracts_leading_inline_code_when_exact_answer_requested():
    prompt = "What exact two characters are missing? Output exactly those characters."
    answer = "`[]`\n\n```js\nuseEffect(() => {}, []);\n```"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "[]"


def test_strips_parentheses_when_method_name_requested():
    prompt = "What built-in dictionary method should be used? Give just the method name."
    assert clean_answer("get()", "general", prompt=prompt) == "get"


def test_extracts_keyword_from_code_debugging_exact_prompt():
    prompt = "What keyword should be added? Reply with only the keyword."
    answer = "```python\ncount = 0\n\ndef increment():\n    global count\n    count += 1\n```\n\nThe keyword is **`global`**."
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "global"


def test_extracts_keyword_from_unfenced_corrected_code():
    prompt = "What specific keyword must be placed before count inside the function?"
    answer = "count = 0\n\ndef increment():\n    global count\n    count += 1"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "global"


def test_extracts_numeric_literal_from_code_debugging_exact_prompt():
    prompt = "What numeric literal should be used? Reply with exactly the literal."
    assert clean_answer("double result = 5 / 2.0;", "code_debugging", prompt=prompt) == "2.0"


def test_extracts_operation_from_code_debugging_exact_prompt():
    prompt = "What exact operation should be applied to arr.length? Reply with only the operation."
    assert clean_answer("let lastItem = arr[arr.length - 1];", "code_debugging", prompt=prompt) == "- 1"


def test_does_not_treat_function_named_add_as_requested_operation():
    prompt = "What is the syntax error? Output the corrected first line only: def add(a, b) return a + b"
    answer = "def add(a, b):\n    return a + b"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "def add(a, b):"


def test_extracts_missing_increment_operation():
    prompt = "What operation is missing inside the loop body?"
    answer = "while (i < 10) { console.log(i); i++; }"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "i++"


def test_recovers_loop_variable_when_model_returns_only_increment_operator():
    prompt = "The code while(i < 10) freezes. What operation is missing inside the loop body?"
    assert clean_answer("++", "code_debugging", prompt=prompt) == "i++"
    assert clean_answer("+", "code_debugging", prompt=prompt) == "i++"


def test_extracts_ignored_css_property():
    prompt = "Which CSS property is effectively ignored?"
    answer = ".box {\n  display: flex;\n  /* float: left; is ignored */\n}"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "float"


def test_extracts_requested_corrected_first_line_from_explanation():
    prompt = "What is the syntax error? Output the corrected first line only."
    answer = "The colon is missing.\n\nCorrected first line:\n```python\ndef add(a, b):\n```"
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "def add(a, b):"


def test_extracts_inline_ignored_property():
    prompt = "Which property is effectively ignored?"
    answer = "The `float` property is effectively ignored."
    assert clean_answer(answer, "code_debugging", prompt=prompt) == "float"


def test_extracts_two_character_fragment_from_code_debugging_exact_prompt():
    prompt = "What exact two characters are missing from this React hook? Reply with exactly those two characters."
    assert clean_answer("useEffect(() => { fetchUser(); }, []);", "code_debugging", prompt=prompt) == "[]"


def test_formats_hhmm_time_when_requested():
    prompt = "Use HH:MM PM format."
    assert clean_answer("4:20 PM", "math", prompt=prompt) == "04:20 PM"


def test_strips_fence_without_language_tag():
    assert clean_answer("```\nSELECT 1;\n```", "code_generation") == "SELECT 1;"


def test_keeps_fences_when_answer_has_surrounding_text():
    mixed = "Here is the fix:\n```python\nx = 1\n```"
    assert clean_answer(mixed, "code_debugging") == mixed


def test_normalizes_unicode_spaces():
    assert clean_answer("reduced memory by 30 %", "summarization") == "reduced memory by 30 %"


def test_strips_answer_prefix_and_quotes():
    assert clean_answer("Answer: 'Canberra'", "factual") == "Canberra"


def test_strips_closed_think_block():
    text = "<think>Let me reason about this.</think>\nCanberra"
    assert clean_answer(text, "factual") == "Canberra"


def test_unclosed_think_block_truncates_to_empty():
    text = "<think>We need answer user. They want fix bug so function"
    assert clean_answer(text, "code_debugging") == ""


def test_sentiment_prose_normalizes_to_label_without_grabbing_negated_label():
    assert clean_answer("The sentiment is positive.", "sentiment") == "positive"
    tricky = "It is not positive; the sentiment is negative."
    assert clean_answer(tricky, "sentiment") == tricky


def test_sentiment_explanation_is_preserved_when_requested():
    prompt = "Classify the sentiment and explain why: The app keeps crashing."
    answer = "Negative: the user reports repeated crashes."
    assert clean_answer(answer, "sentiment", prompt=prompt) == answer


def test_sentiment_json_is_preserved_when_requested():
    prompt = "Return ONLY a valid JSON object mapping each review number to the classification."
    answer = '{"1":"Negative","2":"Neutral","3":"Positive"}'
    assert clean_answer(answer, "sentiment", prompt=prompt) == answer


def test_reasoning_is_preserved_when_work_is_requested():
    prompt = "Show your work: A car travels 120 km in 2 hours. What is its speed?"
    answer = "120 / 2 = 60.\nFINAL ANSWER: 60 km/h"
    assert clean_answer(answer, "math", prompt=prompt) == answer
