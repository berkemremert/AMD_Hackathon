// Package compress trims scored input tokens from task prompts before they
// are sent to the API. The task's own text is billed as prompt tokens, so
// boilerplate politeness and low-content passage sentences cost real score.
// Compression happens in Go (free); the risk is meaning loss, which only the
// live judge can measure — hence the whole package sits behind the
// PROMPT_COMPRESS level knob and ships disabled.
package compress

import (
	"regexp"
	"sort"
	"strings"
)

// Level semantics:
//
//	0 — off (identity)
//	1 — strip politeness/boilerplate wrappers, normalize whitespace
//	2 — level 1 + extractively trim long summarization passages
const (
	LevelOff        = 0
	LevelBoilerplate = 1
	LevelExtractive  = 2
)

var (
	// Politeness/filler openers that carry no task content. Applied
	// repeatedly so stacked openers ("Could you please kindly ...") all go.
	rePolitePrefix = regexp.MustCompile(`(?i)^\s*(?:please|kindly|could you(?: please)?|can you(?: please)?|would you(?: please)?|i would like you to|i'd like you to|i want you to|i need you to|your task is to|you are asked to)[,\s]+`)
	// Trailing sign-offs.
	rePoliteSuffix = regexp.MustCompile(`(?i)[\s]*(?:thank you[.!]?|thanks[.!]?)\s*$`)
	// Mid-sentence "please" as a standalone word.
	rePleaseWord = regexp.MustCompile(`(?i)\bplease\s+`)

	reSpaces   = regexp.MustCompile(`[ \t]+`)
	reNewlines = regexp.MustCompile(`\n{3,}`)

	reSentence = regexp.MustCompile(`[^.!?]+[.!?]?`)
	reWord     = regexp.MustCompile(`[a-zA-Z']+`)
)

// Prompt compresses a task prompt at the given level. isSummarize marks
// prompts whose long passage may be extractively trimmed at level 2.
func Prompt(level int, isSummarize bool, prompt string) string {
	if level <= LevelOff {
		return prompt
	}
	out := stripBoilerplate(prompt)
	if level >= LevelExtractive && isSummarize {
		out = trimPassage(out)
	}
	// Never return something degenerate — compression must not lose the task.
	if len(strings.TrimSpace(out)) < 12 {
		return prompt
	}
	return out
}

func stripBoilerplate(s string) string {
	// Uppercase the new sentence start after each prefix strip.
	for {
		next := rePolitePrefix.ReplaceAllString(s, "")
		if next == s {
			break
		}
		s = upperFirst(strings.TrimSpace(next))
	}
	s = rePoliteSuffix.ReplaceAllString(s, "")
	s = rePleaseWord.ReplaceAllString(s, "")
	s = reSpaces.ReplaceAllString(s, " ")
	s = reNewlines.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}

// trimPassage extractively shortens the passage part of "instruction: passage"
// summarization prompts. Sentences are scored by lexical centrality (shared
// content words with the rest of the passage); the least central are dropped
// from the middle until the passage fits the budget. The instruction and the
// lead sentence are always kept.
const passageBudget = 700 // chars — roughly halves a long passage

func trimPassage(s string) string {
	instr, passage := splitInstruction(s)
	if len(passage) <= passageBudget {
		return s
	}
	sents := reSentence.FindAllString(passage, -1)
	if len(sents) < 3 {
		return s
	}
	// centrality is O(sentences²); on a pathologically long passage that would
	// burn wall-clock. Above a sane bound, take a cheap lead-N slice instead of
	// scoring — a huge passage is exactly where extractive trimming matters
	// most, but not at quadratic cost.
	const maxScored = 300
	if len(sents) > maxScored {
		var b strings.Builder
		for i := 0; i < maxScored && b.Len() < passageBudget; i++ {
			b.WriteString(strings.TrimSpace(sents[i]))
			b.WriteByte(' ')
		}
		lead := strings.TrimSpace(b.String())
		if instr != "" {
			return instr + ": " + lead
		}
		return lead
	}
	scores := centrality(sents)
	// Rank middle sentences (lead stays) by score, drop lowest until we fit.
	type ranked struct {
		idx   int
		score float64
	}
	order := make([]ranked, 0, len(sents)-1)
	for i := 1; i < len(sents); i++ {
		order = append(order, ranked{i, scores[i]})
	}
	sort.Slice(order, func(a, b int) bool { return order[a].score < order[b].score })
	drop := map[int]bool{}
	total := len(passage)
	for _, r := range order {
		if total <= passageBudget {
			break
		}
		drop[r.idx] = true
		total -= len(sents[r.idx])
	}
	var kept []string
	for i, sent := range sents {
		if !drop[i] {
			kept = append(kept, strings.TrimSpace(sent))
		}
	}
	if len(kept) < 2 {
		return s // over-aggressive → keep the original
	}
	if instr != "" {
		return instr + ": " + strings.Join(kept, " ")
	}
	return strings.Join(kept, " ")
}

// splitInstruction separates "Summarise the following in 25 words: <passage>"
// into instruction and passage at the first colon in the opening stretch.
func splitInstruction(s string) (string, string) {
	if i := strings.Index(s, ":"); i > 0 && i < 120 {
		return strings.TrimSpace(s[:i]), strings.TrimSpace(s[i+1:])
	}
	return "", s
}

// centrality scores each sentence by content-word overlap with the others.
func centrality(sents []string) []float64 {
	bags := make([]map[string]bool, len(sents))
	for i, s := range sents {
		bags[i] = map[string]bool{}
		for _, w := range reWord.FindAllString(strings.ToLower(s), -1) {
			if len(w) > 3 && !stopword[w] {
				bags[i][w] = true
			}
		}
	}
	scores := make([]float64, len(sents))
	for i := range sents {
		for j := range sents {
			if i == j {
				continue
			}
			shared := 0
			for w := range bags[i] {
				if bags[j][w] {
					shared++
				}
			}
			scores[i] += float64(shared)
		}
		if n := len(bags[i]); n > 0 {
			scores[i] /= float64(n) // normalize so long sentences don't dominate
		}
	}
	return scores
}

var stopword = map[string]bool{
	"that": true, "this": true, "with": true, "from": true, "they": true,
	"have": true, "were": true, "been": true, "their": true, "which": true,
	"would": true, "could": true, "should": true, "about": true, "after": true,
	"while": true, "though": true, "than": true, "then": true, "them": true,
	"there": true, "these": true, "those": true, "will": true, "into": true,
}

func upperFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}
