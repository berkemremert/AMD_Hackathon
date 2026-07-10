package classify

import (
	"strings"
	"testing"
)

func TestClassifyNeverPanics(t *testing.T) {
	inputs := []string{
		"", " ", "\n\n", "?", "\x00\x01",
		strings.Repeat("A", 100000),
		strings.Repeat("code function bug ", 20000),
		"🎲 unicode 知 の " + strings.Repeat("的", 2000),
		strings.Repeat("+ ", 5000),
	}
	for _, in := range inputs {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Classify panicked on %.20q: %v", in, r)
				}
			}()
			Classify(in)
			ClassifyScored(in)
		}()
	}
}
