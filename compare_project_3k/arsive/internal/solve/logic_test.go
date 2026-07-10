package solve

import (
	"regexp"
	"strings"
	"testing"
)

func TestSolveOrdering(t *testing.T) {
	solved := []struct {
		name, prompt string
		wantContains string
	}{
		{
			"logic-3 race",
			"Four runners finished a race. Dana finished before Eli but after Fay. Gil finished last. Who won the race?",
			"Fay",
		},
		{
			"simple three-way order",
			"Xavier, Yolanda, and Zack ran. Xavier finished before Yolanda. Yolanda finished before Zack. What is the order?",
			"Xavier, Yolanda, Zack",
		},
		{
			"who is last",
			"Anna, Bella, and Clara raced. Anna finished before Bella. Bella finished before Clara. Who finished last?",
			"Clara",
		},
	}
	for _, c := range solved {
		got, ok := SolveOrdering(c.prompt)
		if !ok {
			t.Errorf("%s: expected solve, got defer", c.name)
			continue
		}
		if !strings.Contains(got, c.wantContains) {
			t.Errorf("%s: got %q, want to contain %q", c.name, got, c.wantContains)
		}
	}

	deferred := []struct{ name, prompt string }{
		{"logic-1 seating negation+adjacency", "Three friends — Alice, Bob, and Carol — sit in a row of three seats. Alice does not sit on the left. Bob sits immediately to the right of Carol. Who sits in which seat?"},
		{"lh-3 positional offset", "In a five-person race: Kaya finished first. Lale finished last. Mert finished exactly two places ahead of Nil. Okan finished immediately behind Mert. Give the full finishing order."},
		{"knights-knaves not ordering", "On an island, knights always tell the truth and knaves always lie. A says: 'We are both knaves.' What are A and B?"},
		{"ambiguous incomplete", "Alice, Bob, and Carol ran. Alice finished before Bob. Who won the race?"},
		{"contradiction", "Sam, Tom, and Uma raced. Sam finished before Tom. Tom finished before Uma. Uma finished before Sam. Who won?"},
	}
	for _, c := range deferred {
		if got, ok := SolveOrdering(c.prompt); ok {
			t.Errorf("%s: expected defer, got solve %q", c.name, got)
		}
	}
}

func TestSolveSyllogism(t *testing.T) {
	if got, ok := SolveSyllogism("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no."); !ok {
		t.Errorf("logic-6: expected solve, got defer")
	} else if !regexp.MustCompile(`(?i)^yes`).MatchString(got) {
		t.Errorf("logic-6: got %q, want a Yes answer", got)
	}

	if got, ok := SolveSyllogism("All cats are mammals. All mammals are animals. Are all cats animals?"); !ok || !strings.Contains(strings.ToLower(got), "yes") {
		t.Errorf("cats: got %q ok=%v, want Yes", got, ok)
	}

	deferred := []struct{ name, prompt string }{
		{"non-universal some", "Some cats are black. All black things are dark. Are all cats dark?"},
		{"no chain", "All cats are mammals. All dogs are mammals. Are all cats dogs?"},
		{"single premise", "All cats are mammals. Are all cats animals?"},
		{"not a syllogism", "What is the capital of France?"},
	}
	for _, c := range deferred {
		if got, ok := SolveSyllogism(c.prompt); ok {
			t.Errorf("%s: expected defer, got solve %q", c.name, got)
		}
	}
}
