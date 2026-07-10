from frugalrouter.task_classifier import classify_prompt


def test_date_question_is_not_ner_without_extraction_language():
    assert classify_prompt("What is the date format used in ISO 8601?") == "factual"


def test_entity_extraction_prompt_is_ner():
    assert classify_prompt("Extract people, organizations, locations, and dates from this paragraph.") == "ner"


def test_code_debugging_takes_priority_over_code_generation():
    assert classify_prompt("Debug this Python function and provide corrected code.") == "code_debugging"


def test_natural_sentiment_prompts_are_classified():
    assert classify_prompt("Evaluate the sentiment of this text: I loved it.") == "sentiment"
    assert classify_prompt("Analyze the tone: I am completely obsessed!") == "sentiment"


def test_natural_ner_prompts_are_classified():
    assert classify_prompt("Pull out all the email addresses from this chunk of text.") == "ner"
    assert classify_prompt("Extract the phone number from this transcript.") == "ner"
    assert classify_prompt("Which chemical compounds are explicitly formulaic in this text?") == "ner"


def test_natural_math_prompts_are_classified():
    assert classify_prompt("Multiply 15 by 12 and give the answer.") == "math"
    assert classify_prompt("What is the remainder when 100 is divided by 7?") == "math"
    assert classify_prompt("What time is it completely done? Use HH:MM PM format.") == "math"


def test_natural_code_prompts_are_classified():
    assert classify_prompt("What specific keyword is missing in this async JavaScript function?") == "code_debugging"
    assert classify_prompt("What built-in dictionary method avoids a KeyError?") == "code_debugging"
    assert classify_prompt(
        "This loop freezes: while(i < 10) { console.log(i); }. What operation is missing inside the loop body?"
    ) == "code_debugging"
    assert classify_prompt("Write a regex that matches a literal dollar sign.") == "code_generation"
    assert classify_prompt("Write a Python list comprehension for even squares.") == "code_generation"


def test_logic_cues_without_exactly_one_word_false_positive():
    assert classify_prompt("Premise 1: All blorps are flomps. Are all blorps definitely glarms?") == "logic"
    assert classify_prompt("Who painted Guernica? Provide exactly one word: the artist's last name.") == "factual"

