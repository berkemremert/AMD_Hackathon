package router

import (
	"strings"
	"testing"

	"tokenrouter/internal/classify"
)

func TestMessagesMergeSystem(t *testing.T) {
	merged := &Router{opt: Options{MergeSystem: true}}
	split := &Router{}

	m := merged.messages("SYS", "USER")
	if len(m) != 1 || m[0].Role != "user" {
		t.Fatalf("merge on: want one user message, got %+v", m)
	}
	if !strings.Contains(m[0].Content, "SYS") || !strings.Contains(m[0].Content, "USER") {
		t.Errorf("merged message lost content: %q", m[0].Content)
	}
	if !strings.HasPrefix(m[0].Content, "SYS") {
		t.Errorf("instruction must lead the merged message: %q", m[0].Content)
	}

	s := split.messages("SYS", "USER")
	if len(s) != 2 || s[0].Role != "system" || s[1].Role != "user" {
		t.Fatalf("merge off: want system+user, got %+v", s)
	}
}

func TestGrammarForSentimentOnly(t *testing.T) {
	if grammarFor(classify.Sentiment) == nil {
		t.Error("sentiment must have a grammar")
	}
	for _, cat := range []classify.Category{
		classify.Factual, classify.Math, classify.Logic, classify.NER,
		classify.Summarize, classify.CodeGen, classify.CodeDebug,
	} {
		if grammarFor(cat) != nil {
			t.Errorf("%s must not be grammar-constrained", cat)
		}
	}
	rf := grammarFor(classify.Sentiment)
	if rf["type"] != "grammar" || rf["grammar"] == "" {
		t.Errorf("unexpected response_format payload: %v", rf)
	}
}

func TestCompressPassthroughWhenOff(t *testing.T) {
	r := &Router{} // PromptCompress: 0
	in := "Please   answer  this."
	if got := r.compress(classify.Factual, in); got != in {
		t.Errorf("compress off must be identity, got %q", got)
	}
	on := &Router{opt: Options{PromptCompress: 1}}
	if got := on.compress(classify.Factual, in); strings.Contains(strings.ToLower(got), "please") {
		t.Errorf("compress on must strip boilerplate, got %q", got)
	}
}
