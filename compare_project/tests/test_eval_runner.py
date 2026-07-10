from frugalrouter.eval_runner import grade_answer, summarize


def test_grade_answer_exact_match_normalizes_whitespace_and_case():
    grade = grade_answer("  ready  ", {"expected_exact": "READY"})

    assert grade["passed"] is True


def test_grade_answer_contains_all_reports_failure():
    grade = grade_answer("Lisa joined AMD.", {"expected_contains_all": ["Lisa", "AMD", "Austin"]})

    assert grade["passed"] is False
    assert "Austin" in grade["reason"]


def test_grade_answer_max_words_fails_when_too_long():
    spec = {"expected_contains_any": ["bike"], "expected_max_words": 5}
    grade = grade_answer("The city expanded its bike lane network significantly this year", spec)

    assert grade["passed"] is False
    assert "over_word_limit" in grade["reason"]


def test_grade_answer_max_words_passes_when_within_limit():
    spec = {"expected_contains_any": ["bike"], "expected_max_words": 8}
    grade = grade_answer("Council expands bike lanes", spec)

    assert grade["passed"] is True


def test_grade_answer_max_words_requires_content_too():
    spec = {"expected_contains_any": ["bike"], "expected_max_words": 8}
    grade = grade_answer("Council expands transit options", spec)

    assert grade["passed"] is False


def test_summarize_counts_tokens_and_remote_rate():
    report = summarize(
        [
            {
                "passed": True,
                "used_remote": False,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "route": "local",
                "detected_category": "math",
                "task_id": "t1",
            },
            {
                "passed": False,
                "used_remote": True,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "route": "remote",
                "detected_category": "factual",
                "task_id": "t2",
            },
        ],
        elapsed_seconds=1.2345,
    )

    assert report["accuracy"] == 0.5
    assert report["remote_calls"] == 1
    assert report["total_tokens"] == 15

