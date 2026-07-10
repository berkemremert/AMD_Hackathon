package compress

import (
	"strings"
	"testing"
)

func TestPromptNeverPanics(t *testing.T) {
	inputs := []string{
		"", " ", "\n\n\n", ":", "please",
		strings.Repeat("A", 200000),
		strings.Repeat("Please kindly. ", 10000),
		"Summarize: " + strings.Repeat("sentence one. ", 20000),
		"🎲 " + strings.Repeat("的", 5000),
	}
	for _, lvl := range []int{0, 1, 2} {
		for _, in := range inputs {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("Prompt(%d) panicked on %.20q: %v", lvl, in, r)
					}
				}()
				out := Prompt(lvl, true, in)
				// Compression must never lose everything from a non-empty prompt.
				if len(strings.TrimSpace(in)) >= 12 && strings.TrimSpace(out) == "" {
					t.Errorf("Prompt(%d) emptied a non-empty prompt %.20q", lvl, in)
				}
			}()
		}
	}
}
