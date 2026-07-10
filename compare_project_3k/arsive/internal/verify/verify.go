package verify

import (
	"regexp"
	"strings"

	"tokenrouter/internal/classify"
)

var (
	reNERType   = regexp.MustCompile(`(?i)\b(person|organi[sz]ation|location|date|place|company|org)\b`)
	reNumeric   = regexp.MustCompile(`-?\d[\d,]*\.?\d*`)
	reSentWord  = regexp.MustCompile(`(?i)\b(positive|negative|neutral|mixed)\b`)
	reRefusal   = regexp.MustCompile(`(?i)(i (can'?t|cannot|am unable)|as an ai|i don'?t know|i'?m not sure)`)
	reSentSplit = regexp.MustCompile(`[.!?]+\s`)
	reWordLimit = regexp.MustCompile(`(?i)in (\d+) words? or (?:less|fewer)`)
	reOneSent   = regexp.MustCompile(`(?i)in (?:one|a single|1) sentence`)
)

// Check runs cheap, category-specific sanity checks on a candidate answer.
// It gates escalation: answers that fail here get one local retry, then go
// to Fireworks. Deliberately conservative — a false "ok" costs accuracy,
// a false "fail" costs tokens.
func Check(cat classify.Category, prompt, answer string) bool {
	a := strings.TrimSpace(answer)
	if a == "" || reRefusal.MatchString(a) {
		return false
	}
	switch cat {
	case classify.Sentiment:
		return reSentWord.MatchString(a)
	case classify.Math:
		// Post-processing reduces a proper answer to its final line; a long
		// answer here means the model never emitted 'Answer: <number>'
		// (usually a truncated step dump) and must not pass.
		return reNumeric.MatchString(a) && len(a) <= 60
	case classify.NER:
		// Expect at least one typed entity mention; separator style varies
		// (em dash, colon, parens), so key off the type words themselves.
		return reNERType.MatchString(a)
	case classify.Summarize:
		return checkSummaryConstraints(prompt, a)
	case classify.Logic, classify.Factual:
		return len(a) >= 2
	case classify.CodeGen, classify.CodeDebug:
		// Real validation (syntax check / execution) happens in the router
		// via the solve package; here we only reject obvious non-code.
		return len(a) >= 10
	}
	return true
}

func checkSummaryConstraints(prompt, answer string) bool {
	if reOneSent.MatchString(prompt) {
		if sentenceCount(answer) > 1 {
			return false
		}
	}
	if m := reWordLimit.FindStringSubmatch(prompt); m != nil {
		limit := atoiSafe(m[1])
		if limit > 0 && wordCount(answer) > limit {
			return false
		}
	}
	return true
}

func sentenceCount(s string) int {
	n := len(reSentSplit.FindAllString(strings.TrimSpace(s), -1)) + 1
	return n
}

func wordCount(s string) int {
	return len(strings.Fields(s))
}

func atoiSafe(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + int(c-'0')
	}
	return n
}
