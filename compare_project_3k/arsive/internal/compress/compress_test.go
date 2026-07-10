package compress

import (
	"strings"
	"testing"
)

func TestLevelOffIsIdentity(t *testing.T) {
	in := "Please kindly summarize this.   Thanks!"
	if got := Prompt(LevelOff, false, in); got != in {
		t.Errorf("level 0 must not touch the prompt, got %q", got)
	}
}

func TestBoilerplateStrip(t *testing.T) {
	in := "Could you please kindly tell me what the capital of France is? Thank you."
	got := Prompt(LevelBoilerplate, false, in)
	if strings.Contains(strings.ToLower(got), "please") ||
		strings.Contains(strings.ToLower(got), "could you") ||
		strings.Contains(strings.ToLower(got), "thank") {
		t.Errorf("boilerplate survived: %q", got)
	}
	// The actual question must survive intact.
	if !strings.Contains(got, "capital of France") {
		t.Errorf("task content lost: %q", got)
	}
}

func TestWhitespaceNormalization(t *testing.T) {
	got := Prompt(LevelBoilerplate, false, "What   is\t 2+2?\n\n\n\nAnswer briefly.")
	if strings.Contains(got, "  ") || strings.Contains(got, "\n\n\n") {
		t.Errorf("whitespace not collapsed: %q", got)
	}
}

func TestNeverDegenerate(t *testing.T) {
	// A prompt that is nothing but boilerplate must come back unchanged
	// rather than empty.
	in := "Please. Thanks!"
	if got := Prompt(LevelBoilerplate, false, in); got != in {
		t.Errorf("degenerate compression must fall back to the original, got %q", got)
	}
}

func TestExtractiveTrimShortens(t *testing.T) {
	passage := strings.Repeat("The council debated the new transit budget through the evening session. ", 8) +
		"Officials confirmed the subway signal upgrade is funded for next year. " +
		strings.Repeat("Several unrelated procedural motions about committee seating followed. ", 8)
	in := "Summarise the following in 25 words or fewer: " + passage
	got := Prompt(LevelExtractive, true, in)
	if len(got) >= len(in) {
		t.Errorf("expected a shorter prompt, got %d >= %d chars", len(got), len(in))
	}
	// The instruction must survive verbatim-ish.
	if !strings.Contains(got, "25 words") {
		t.Errorf("instruction lost: %q", got)
	}
	// The lead sentence always survives.
	if !strings.Contains(got, "council debated") {
		t.Errorf("lead sentence lost: %q", got)
	}
}

func TestExtractiveLeavesShortPassagesAlone(t *testing.T) {
	in := "Summarise the following in one sentence: The cat sat on the mat. The dog barked."
	got := Prompt(LevelExtractive, true, in)
	if !strings.Contains(got, "cat sat") || !strings.Contains(got, "dog barked") {
		t.Errorf("short passage must not be trimmed: %q", got)
	}
}

func TestExtractiveOnlyForSummarize(t *testing.T) {
	long := "Explain this code: " + strings.Repeat("x = compute(x). ", 100)
	got := Prompt(LevelExtractive, false, long)
	// Non-summarize prompts get level-1 treatment only (no sentence dropping).
	if strings.Count(got, "compute") != 100 {
		t.Errorf("non-summarize passage was trimmed: %d occurrences", strings.Count(got, "compute"))
	}
}
