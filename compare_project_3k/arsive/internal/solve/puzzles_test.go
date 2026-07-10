package solve

import (
	"strings"
	"testing"
)

// --- knights-and-knaves ---

func TestSolveKnightsBasic(t *testing.T) {
	// eval/hard.json lh-2 — unique solution: A knight, B knave, C knave.
	prompt := "On an island of knights (always truthful) and knaves (always lying): " +
		"A says 'B is a knave.' B says 'A and C are the same type.' C says 'B is a knight.' " +
		"Determine what each of A, B, and C is."
	ans, ok := SolveKnights(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	for _, want := range []string{"A is a knight", "B is a knave", "C is a knave"} {
		if !strings.Contains(ans, want) {
			t.Errorf("answer %q missing %q", ans, want)
		}
	}
}

func TestSolveKnightsTwoPeople(t *testing.T) {
	prompt := "Knights always tell the truth and knaves always lie. " +
		"Zed says 'I am a knight.' Ash says 'Zed is a knave.' What are Zed and Ash?"
	// Zed's claim is consistent either way for Zed alone... but Ash constrains:
	// if Ash is a knight, Zed is a knave — Zed (knave) saying "I am a knight" is
	// a lie: consistent. If Ash is a knave, Zed is a knight — "I am a knight"
	// true: consistent. Two solutions → must defer.
	if ans, ok := SolveKnights(prompt); ok {
		t.Fatalf("ambiguous puzzle must defer, got %q", ans)
	}
}

func TestSolveKnightsWeBoth(t *testing.T) {
	// eval logic-2 shape: single statement, silent second participant.
	prompt := "On an island, knights always tell the truth and knaves always lie. " +
		"You meet two people. A says: 'We are both knaves.' What are A and B?"
	ans, ok := SolveKnights(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "A is a knave") || !strings.Contains(ans, "B is a knight") {
		t.Errorf("wrong answer: %q", ans)
	}
}

func TestSolveKnightsDefers(t *testing.T) {
	cases := []string{
		// Unparseable statement form → defer.
		"Knights tell the truth, knaves lie. A says 'if B is a knight then C is a knave.' B says 'A is a knave.' What are they?",
		// No knight/knave framing at all.
		"A says 'B is a knave.' B says 'A is a knight.' Who is lying?",
		// A "says" without a captured quote.
		"Knights and knaves: A says that B is lying. B says 'A is a knave.' Who is what?",
	}
	for _, p := range cases {
		if ans, ok := SolveKnights(p); ok {
			t.Errorf("expected defer for %q, got %q", p, ans)
		}
	}
}

// --- zebra grids ---

func TestSolveZebraHard1(t *testing.T) {
	// eval/hard.json lh-1 — full grid underdetermined (cat/bird swap), but the
	// queried cells are unique: Ana owns the fish, Ana's house is red.
	prompt := "Four friends — Ana, Ben, Cem, and Deniz — each own one pet (cat, dog, fish, bird) " +
		"and live in different colored houses (red, blue, green, yellow). Ana owns neither the cat " +
		"nor the dog. The fish owner lives in the red house. Ben lives in the blue house. Cem owns " +
		"the dog. Deniz lives in neither the red nor the green house. Who owns the fish, and what " +
		"color is Ana's house?"
	ans, ok := SolveZebra(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "Ana owns the fish") || !strings.Contains(ans, "Ana's house is red") {
		t.Errorf("wrong answer: %q", ans)
	}
}

func TestSolveZebraDefers(t *testing.T) {
	cases := []string{
		// Unparsed clue ("next to") → defer.
		"Three friends — Ana, Ben, Cem — own pets (cat, dog, fish) and live in houses (red, blue, green). Ana lives next to the red house. Cem owns the dog. Who owns the fish?",
		// Only one domain declared.
		"Ana, Ben, and Cem own pets (cat, dog, fish). Cem owns the dog. Ana does not own the cat. Who owns the fish?",
		// Queried cell genuinely ambiguous.
		"Three friends — Ana, Ben, Cem — own pets (cat, dog, fish) and live in houses (red, blue, green). Cem owns the dog. Cem lives in the red house. Who owns the fish?",
	}
	for _, p := range cases {
		if ans, ok := SolveZebra(p); ok {
			t.Errorf("expected defer for %q, got %q", p, ans)
		}
	}
}

// --- positional races ---

func TestSolvePositionsHard3(t *testing.T) {
	// eval/hard.json lh-3.
	prompt := "In a five-person race with positions 1 through 5: Kaya finished first. " +
		"Lale finished last. Mert finished exactly two places ahead of Nil. " +
		"Okan finished immediately behind Mert. Give the full finishing order."
	ans, ok := SolvePositions(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "Kaya, Mert, Okan, Nil, Lale") {
		t.Errorf("wrong order: %q", ans)
	}
}

func TestSolvePositionsWhoWon(t *testing.T) {
	// Order not fully determined (Rita/Sam ambiguous), but position 1 is.
	prompt := "Four runners raced: Paco, Quin, Rita, and Sam. Paco finished exactly three places ahead of Quin. " +
		"Rita finished behind Paco. Sam finished behind Paco. Who finished first?"
	ans, ok := SolvePositions(prompt)
	if !ok {
		t.Fatal("expected a solve")
	}
	if !strings.Contains(ans, "Paco finished first") {
		t.Errorf("wrong answer: %q", ans)
	}
}

func TestSolvePositionsDefers(t *testing.T) {
	cases := []string{
		// Unparseable clue about a participant.
		"Ana finished first. Ben was not last. Cem finished immediately behind Ana. What is the finishing order?",
		// Ambiguous full order.
		"Ana finished first. Ben finished behind Ana. Cem finished behind Ana. Give the full finishing order.",
	}
	for _, p := range cases {
		if ans, ok := SolvePositions(p); ok {
			t.Errorf("expected defer for %q, got %q", p, ans)
		}
	}
}
