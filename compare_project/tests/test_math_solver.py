from frugalrouter.math_solver import solve_simple_math


def test_solves_parenthesized_expression():
    assert solve_simple_math("Calculate (8 + 4) * 3. Return only the number.") == (
        "36",
        "direct_arithmetic_expression",
    )


def test_solves_percentage_of_pattern():
    assert solve_simple_math("What is 12.5% of 80? Return only the number.") == ("10", "percentage_pattern")


def test_ignores_non_math_prompt_with_numbers():
    assert solve_simple_math("Explain what happened in 2024 and 2025.") is None


def test_rejects_evaluable_span_inside_non_math_question():
    assert solve_simple_math("What is 12/25 known as in the US holiday calendar?") is None


def test_rejects_chained_percentages():
    assert solve_simple_math("What is 20% of 50% of 200?") is None


def test_rejects_percentage_embedded_in_word_problem():
    assert solve_simple_math("What is 15% of 200, plus 10? Return only the number.") is None
    assert solve_simple_math("A shop wants to discount 40 by 25 percent and then add 5 for shipping.") is None


def test_still_solves_pure_percentage_forms():
    assert solve_simple_math("Increase 80 by 12.5 percent. Return only the number.") == ("90", "percentage_pattern")
    assert solve_simple_math("30 is what percent of 120?") == ("25", "percentage_pattern")

