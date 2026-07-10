import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from solvers import solve_arithmetic, solve_logic


class TestLogic(unittest.TestCase):
    def test_practice07_who_owns(self):
        p = ("Three friends - Alice, Bob, and Carol - each own a different "
             "pet: a cat, a dog, and a bird. Alice does not own the cat or "
             "the dog. Bob does not own the bird. Who owns the bird?")
        self.assertEqual(solve_logic(p), "Alice owns the bird.")

    def test_direct_assignment_query_person(self):
        p = ("Sam, Jo, and Lee each own a different pet: cat, dog, bird. "
             "Sam owns the dog. Jo does not own the cat. What does Lee own?")
        # Sam=dog; Jo not cat -> Jo=bird; Lee=cat.
        self.assertEqual(solve_logic(p), "Lee owns the cat.")

    def test_parenthetical_domain(self):
        p = ("Ann and Ben each own a different pet (cat, dog). "
             "Ann does not own the cat. Who owns the cat?")
        self.assertEqual(solve_logic(p), "Ben owns the cat.")

    # --- Must defer (return None) ---
    def test_defer_non_unique(self):
        p = ("Alice, Bob, and Carol each own a different pet: cat, dog, bird. "
             "Alice owns the cat. Who owns the bird?")  # bird/dog ambiguous
        self.assertIsNone(solve_logic(p))

    def test_defer_two_domains_zebra(self):
        p = ("Alice and Bob each own a pet (cat, dog) and a house "
             "(red, blue). Alice owns the cat. Who owns the dog?")
        self.assertIsNone(solve_logic(p))

    def test_defer_non_ownership_relation(self):
        p = ("Alice, Bob, and Carol each like a different color: red, green, "
             "blue. Alice likes red. Who likes blue?")
        self.assertIsNone(solve_logic(p))

    def test_defer_person_count_mismatch(self):
        p = ("Alice and Bob each own a different pet: cat, dog, bird. "
             "Alice owns the cat. Who owns the bird?")
        self.assertIsNone(solve_logic(p))

    def test_defer_unparseable_clue(self):
        p = ("Alice, Bob, and Carol each own a different pet: cat, dog, bird. "
             "The person who owns the cat is taller than Bob. Who owns the bird?")
        self.assertIsNone(solve_logic(p))

    def test_defer_no_domain(self):
        self.assertIsNone(solve_logic("Who is taller, Alice or Bob?"))


class TestArithmetic(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(solve_arithmetic("What is 12 * 34?"), "408")

    def test_parentheses(self):
        self.assertEqual(solve_arithmetic("Calculate (5 + 3) * 2"), "16")

    def test_decimal(self):
        self.assertEqual(solve_arithmetic("Compute 10 / 4"), "2.5")

    def test_subtraction_chain(self):
        self.assertEqual(solve_arithmetic("What is 240 - 36 - 60?"), "144")

    # --- Must defer ---
    def test_defer_word_problem(self):
        self.assertIsNone(solve_arithmetic(
            "A store has 240 items. It sells 15% on Monday. How many remain?"))

    def test_defer_percent_of(self):
        self.assertIsNone(solve_arithmetic("What is 15% of 240?"))

    def test_defer_bare_number(self):
        self.assertIsNone(solve_arithmetic("What is 42?"))

    def test_defer_division_by_zero(self):
        self.assertIsNone(solve_arithmetic("What is 5 / 0?"))

    def test_defer_letters(self):
        self.assertIsNone(solve_arithmetic("What is x + 2?"))


if __name__ == "__main__":
    unittest.main()
