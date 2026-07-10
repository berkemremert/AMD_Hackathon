package classify

import (
	"regexp"
	"strings"
)

type Category string

const (
	Factual   Category = "factual"
	Math      Category = "math"
	Sentiment Category = "sentiment"
	Summarize Category = "summarize"
	NER       Category = "ner"
	CodeDebug Category = "code_debug"
	Logic     Category = "logic"
	CodeGen   Category = "code_gen"
)

var (
	reCodeBlock  = regexp.MustCompile("```|def |class |function |import |#include|console\\.log|System\\.out|func |fn |=>|print\\(")
	reNumber     = regexp.MustCompile(`\d`)
	reArithmetic = regexp.MustCompile(`\d[\d,.]*\s*[-+*/×÷%^]\s*[\d(]`)
	reMathWords  = regexp.MustCompile(`(?i)\b(calculate|compute|how (much|many)|total|percent|percentage|sum of|difference|product of|divided|multiply|average|interest|discount|profit|revenue|cost|price|grow(s|th)?|increase|decrease|remainder|ratio|per (hour|day|week|month|year))\b`)
	reSentiment  = regexp.MustCompile(`(?i)\b(sentiment|positive,? negative|negative,? positive|tone of|emotion(al)? (tone|polarity)|classify.{0,40}(review|tweet|comment|feedback|statement))\b`)
	// Note: bare "in one sentence" is a generic instruction (logic tasks say
	// "explain in one sentence" too) — real summarization tasks carry a verb.
	reSummarize  = regexp.MustCompile(`(?i)\b(summar(y|ize|ise|isation|ization)|condense|tl;?dr|shorten (the|this)|gist|boil .{0,20}down|in a nutshell|key points?|restate|rephrase|in (at most |no more than |under )?\d+ words)\b`)
	reNER        = regexp.MustCompile(`(?i)\b(named entit|entit(y|ies)|extract (and label|all|the)?\s*(people|persons?|names|organi[sz]ations?|locations?|dates?|places)|identify (all |the )?(people|persons?|organi[sz]ations?|locations?|dates?))\b`)
	reDebug      = regexp.MustCompile(`(?i)\b(bug(gy|s)?|debug|fix (the|this|my)|error in|doesn'?t work|not work(ing)?|incorrect(ly)? (output|result)|why does .{0,60}(fail|crash|return)|find (the|and fix))\b`)
	reCodeGen    = regexp.MustCompile(`(?i)\b(write|implement|create|build|develop|generate)\b.{0,60}\b(function|method|class|program|script|code|algorithm|api|regex|sql)\b`)
	reLogic      = regexp.MustCompile(`(?i)\b(puzzle|riddle|constraint|deduce|deduction|logical(ly)?|who (owns|lives|sits|is the|won|finished)|seated|arrange|knights?|knaves?|truth[- ]?teller|liar|always (tell|tells) the truth|always lies?|exactly one|if all .{0,40} then|conclude|in a row|immediately (to the )?(left|right)|far (left|right)|next to (the|each other))\b`)
)

// Classify scores each category with cheap lexical signals and returns the
// best match. Ambiguous prompts fall back to Factual, which is the safest
// default pipeline (plain local answer).
func Classify(prompt string) Category {
	c, _ := ClassifyScored(prompt)
	return c
}

// ClassifyScored also reports the winning score so callers can detect weak
// signals (0 = pure default, 1 = single soft hint) and consult the local
// model instead of trusting a blind fallback.
func ClassifyScored(prompt string) (Category, int) {
	p := strings.ToLower(prompt)
	hasCode := reCodeBlock.MatchString(prompt)

	scores := map[Category]int{}

	if reSentiment.MatchString(p) {
		scores[Sentiment] += 3
	}
	if reSummarize.MatchString(p) {
		scores[Summarize] += 3
	}
	if reNER.MatchString(p) {
		scores[NER] += 3
	}
	if reDebug.MatchString(p) {
		scores[CodeDebug] += 2
		if hasCode {
			scores[CodeDebug] += 3
		}
	}
	if reCodeGen.MatchString(p) {
		scores[CodeGen] += 3
		if hasCode && scores[CodeDebug] > 0 {
			scores[CodeGen]-- // provided code + bug language usually means debugging
		}
	}
	if reLogic.MatchString(p) {
		scores[Logic] += 2
	}
	if reMathWords.MatchString(p) && reNumber.MatchString(p) {
		scores[Math] += 2
	}
	if reArithmetic.MatchString(p) {
		scores[Math] += 2
	}
	// A code block with no other signal is still most likely a code task.
	if hasCode && scores[CodeDebug] == 0 && scores[CodeGen] == 0 {
		scores[CodeDebug]++
	}

	best, bestScore := Factual, 0
	// Deterministic tie-break order: more specific categories win.
	order := []Category{Sentiment, NER, Summarize, CodeDebug, CodeGen, Logic, Math}
	for _, c := range order {
		if scores[c] > bestScore {
			best, bestScore = c, scores[c]
		}
	}
	return best, bestScore
}
