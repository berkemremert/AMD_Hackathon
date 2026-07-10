package solve

import (
	"strings"
	"testing"
)

func TestSolveSingleAssignPractice07(t *testing.T) {
	// The guide's official practice-07.
	prompt := "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. " +
		"Sam does not own the bird. Jo owns the dog. Who owns the cat?"
	ans, ok := SolveSingleAssign(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "Sam owns the cat") {
		t.Errorf("wrong answer: %q", ans)
	}
}

func TestSolveSingleAssignParenthetical(t *testing.T) {
	prompt := "Four coworkers — Ana, Ben, Cem, and Dua — each drink a different beverage (tea, coffee, milk, juice). " +
		"Ana drinks neither the tea nor the milk. Ben has the coffee. Cem does not have the juice. " +
		"Ana does not have the juice. What does Cem drink?"
	// Ben=coffee; Ana ∉ {tea, milk, juice, coffee}... Ana must have juice —
	// contradiction with 'Ana does not have the juice' → no solution → defer.
	if ans, ok := SolveSingleAssign(prompt); ok {
		t.Fatalf("contradictory puzzle must defer, got %q", ans)
	}
}

func TestSolveSingleAssignQueryUnique(t *testing.T) {
	// Full grid ambiguous (Ana/Cem swap tea/milk) but the query is unique.
	prompt := "Three coworkers — Ana, Ben, and Cem — each drink a different beverage (tea, coffee, milk). " +
		"Ben has the coffee. Who owns the coffee?"
	// "Who owns the coffee" uses owns-phrasing on a drink; keep the canonical
	// owns query for the pet variant instead:
	prompt = "Three kids — Ana, Ben, and Cem — each own a different pet (cat, dog, fish). " +
		"Ben owns the dog. Who owns the dog?"
	ans, ok := SolveSingleAssign(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "Ben owns the dog") {
		t.Errorf("wrong answer: %q", ans)
	}
	// But an ambiguous queried cell must defer.
	prompt = "Three kids — Ana, Ben, and Cem — each own a different pet (cat, dog, fish). " +
		"Ben owns the dog. Who owns the cat?"
	if ans, ok := SolveSingleAssign(prompt); ok {
		t.Fatalf("ambiguous query cell must defer, got %q", ans)
	}
}

func TestSolveSingleAssignDefers(t *testing.T) {
	cases := []string{
		// Unparsed clue phrasing ("prefers").
		"Three kids — Ana, Ben, Cem — each own a different pet: cat, dog, fish. Ana prefers to own the cat. Ben owns the dog. Who owns the fish?",
		// Two domains → zebra's territory, not this solver's.
		"Three kids — Ana, Ben, Cem — own pets (cat, dog, fish) and live in houses (red, blue, green). Ben owns the dog. Who owns the cat?",
		// People/value count mismatch.
		"Three kids — Ana, Ben, Cem, and Dua — each own a different pet: cat, dog, fish. Ben owns the dog. Who owns the cat?",
	}
	for _, p := range cases {
		if ans, ok := SolveSingleAssign(p); ok {
			t.Errorf("expected defer for %q, got %q", p, ans)
		}
	}
}
