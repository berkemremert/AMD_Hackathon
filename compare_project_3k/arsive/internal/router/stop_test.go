package router

import (
	"testing"

	"tokenrouter/internal/classify"
)

func TestStopFor(t *testing.T) {
	on := &Router{opt: Options{StopSeqs: true}}
	off := &Router{opt: Options{StopSeqs: false}}

	cases := []struct {
		cat       classify.Category
		wantOnLen int
	}{
		{classify.Sentiment, 1},
		{classify.Factual, 1},
		{classify.Summarize, 1},
		{classify.NER, 1},
		{classify.Math, 0},
		{classify.Logic, 0},
		{classify.CodeGen, 0},
		{classify.CodeDebug, 0},
	}
	for _, c := range cases {
		if got := on.stopFor(c.cat); len(got) != c.wantOnLen {
			t.Errorf("stopFor(%s) enabled = %v, want len %d", c.cat, got, c.wantOnLen)
		}
		if got := off.stopFor(c.cat); got != nil {
			t.Errorf("stopFor(%s) disabled = %v, want nil", c.cat, got)
		}
	}
	// The dangerous "\n" must never be a stop for NER or code.
	for _, c := range []classify.Category{classify.NER, classify.CodeGen, classify.CodeDebug} {
		for _, s := range on.stopFor(c) {
			if s == "\n" {
				t.Errorf("stopFor(%s) contains bare \\n — would truncate the answer", c)
			}
		}
	}
}
