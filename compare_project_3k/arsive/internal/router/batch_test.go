package router

import (
	"testing"

	"tokenrouter/internal/classify"
)

func TestParseBatch(t *testing.T) {
	t.Run("clean", func(t *testing.T) {
		out, ok := parseBatch("1: Positive\n2: Negative\n3: Neutral", 3, classify.Sentiment)
		if !ok || len(out) != 3 || out[0] != "Positive" || out[2] != "Neutral" {
			t.Fatalf("got %v ok=%v", out, ok)
		}
	})
	t.Run("separators", func(t *testing.T) {
		out, ok := parseBatch("1. Paris\n2) Berlin\n3: Rome", 3, classify.Factual)
		if !ok || out[1] != "Berlin" {
			t.Fatalf("got %v ok=%v", out, ok)
		}
	})
	t.Run("multiline answer", func(t *testing.T) {
		out, ok := parseBatch("1: Positive because\nthe tone is warm\n2: Negative", 2, classify.Sentiment)
		if !ok || out[0] != "Positive because\nthe tone is warm" || out[1] != "Negative" {
			t.Fatalf("got %#v ok=%v", out, ok)
		}
	})
	t.Run("missing item fails", func(t *testing.T) {
		if _, ok := parseBatch("1: Positive\n3: Neutral", 3, classify.Sentiment); ok {
			t.Fatal("expected failure on missing item 2")
		}
	})
	t.Run("too few markers fails", func(t *testing.T) {
		if _, ok := parseBatch("just one blob of prose", 3, classify.Sentiment); ok {
			t.Fatal("expected failure with no markers")
		}
	})
	t.Run("math answer-line extraction", func(t *testing.T) {
		// postprocess for Math/Logic pulls the Answer: line; batch reuses it.
		out, ok := parseBatch("1: Reasoning.\nAnswer: 42\n2: Answer: 7", 2, classify.Logic)
		if !ok || out[0] != "42" || out[1] != "7" {
			t.Fatalf("got %#v ok=%v", out, ok)
		}
	})
}

func TestBatchable(t *testing.T) {
	cases := []struct {
		cat    classify.Category
		prompt string
		want   bool
	}{
		{classify.Sentiment, "Great phone!", true},
		{classify.Factual, "Capital of France?", true},
		{classify.Summarize, "Summarize: ...", false},
		{classify.Math, "2+2?", false},
		{classify.CodeGen, "Write fib", false},
		{classify.Sentiment, "Line one\nline two", false}, // multi-line excluded
	}
	for _, c := range cases {
		if got := Batchable(c.cat, c.prompt); got != c.want {
			t.Errorf("Batchable(%s, %.20q) = %v, want %v", c.cat, c.prompt, got, c.want)
		}
	}
}
