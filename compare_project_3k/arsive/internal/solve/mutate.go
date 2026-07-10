package solve

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// Mutation repair (automated-program-repair style): most textbook debug tasks
// hide a single-token bug — an off-by-one, a flipped comparison, a wrong
// operator. Plain code can enumerate every single-edit mutant of the buggy
// snippet and run each against asserts derived from the prompt's own examples.
// Proof rule, in line with the router's golden rule (never answer unproven):
//
//	answer only when the ORIGINAL code fails the asserts (the bug is real)
//	and EXACTLY ONE mutant passes them (the fix is unambiguous);
//	anything else → defer to the model.
//
// Zero tokens on success; on deferral the normal paid path runs unchanged.

type mutation struct {
	code string // full mutated snippet
	old  string // human-readable description parts
	new  string
}

// operator swap table; longer tokens first so "<=" is seen before "<".
// Both strictness flips (< ↔ <=) and direction flips (< ↔ >) are classic
// single-token bugs.
var mutOps = [][2]string{
	{"<=", "<"}, {"<=", ">="}, {">=", ">"}, {">=", "<="},
	{"<", "<="}, {"<", ">"}, {">", ">="}, {">", "<"},
	{"==", "!="}, {"!=", "=="},
	{" and ", " or "}, {" or ", " and "},
	{" + ", " - "}, {" - ", " + "},
}

var (
	reMutInt     = regexp.MustCompile(`\b\d+\b`)
	reMutRange   = regexp.MustCompile(`range\(([^(),]+)\)`)
	reMutRange2  = regexp.MustCompile(`range\(([^(),]+),([^(),]+)\)`)
	rePromptCode = regexp.MustCompile(`(?m)^(?:def |class |import |from )`)
)

const (
	mutMaxMutants = 60
	mutRunTimeout = 3 * time.Second
	mutBudget     = 12 * time.Second // wall-clock cap for the whole attempt
)

// ExtractPromptCode pulls the buggy snippet out of a debug prompt: everything
// from the first def/class/import line to the end (debug prompts put prose
// first, code last). Falls back to a fenced block if one exists.
func ExtractPromptCode(prompt string) string {
	if i := strings.Index(prompt, "```"); i >= 0 {
		return ExtractCode(prompt)
	}
	if loc := rePromptCode.FindStringIndex(prompt); loc != nil {
		return strings.TrimSpace(prompt[loc[0]:])
	}
	return ""
}

// RepairPython tries single-edit mutants of code against the asserts.
// On success it returns the fixed code and a one-line bug description.
func RepairPython(ctx context.Context, code string, asserts []string) (fixed, desc string, ok bool) {
	if code == "" || len(asserts) == 0 {
		return "", "", false
	}
	budget, cancel := context.WithTimeout(ctx, mutBudget)
	defer cancel()

	// The bug must be provable: the original fails its own examples.
	if runPasses(budget, code, asserts) {
		return "", "", false
	}

	var passing []mutation
	seen := map[string]bool{code: true}
	for _, m := range generateMutants(code) {
		if seen[m.code] {
			continue
		}
		seen[m.code] = true
		if budget.Err() != nil {
			return "", "", false // out of time — defer rather than half-search
		}
		if runPasses(budget, m.code, asserts) {
			passing = append(passing, m)
			if len(passing) > 1 {
				return "", "", false // ambiguous fix → defer
			}
		}
	}
	if len(passing) != 1 {
		return "", "", false
	}
	m := passing[0]
	return m.code, fmt.Sprintf("The bug: `%s` should be `%s`.", strings.TrimSpace(m.old), strings.TrimSpace(m.new)), true
}

// generateMutants emits every single-site edit from the mutation table.
func generateMutants(code string) []mutation {
	var out []mutation
	add := func(mutated, old, new string) {
		if mutated != code && len(out) < mutMaxMutants {
			out = append(out, mutation{code: mutated, old: old, new: new})
		}
	}

	// Operator swaps, one occurrence at a time.
	for _, op := range mutOps {
		from, to := op[0], op[1]
		for idx := 0; ; {
			j := strings.Index(code[idx:], from)
			if j < 0 {
				break
			}
			pos := idx + j
			idx = pos + len(from)
			if inStringOrComment(code, pos) {
				continue
			}
			// Skip "<" that is part of "<=" (and friends) so single-char ops
			// don't corrupt their two-char siblings.
			if len(from) == 1 && pos+1 < len(code) && code[pos+1] == '=' {
				continue
			}
			add(code[:pos]+to+code[pos+len(from):], from, to)
		}
	}

	// Integer literal off-by-one, both directions.
	for _, loc := range reMutInt.FindAllStringIndex(code, -1) {
		if inStringOrComment(code, loc[0]) {
			continue
		}
		lit := code[loc[0]:loc[1]]
		n := 0
		fmt.Sscanf(lit, "%d", &n)
		add(code[:loc[0]]+fmt.Sprint(n+1)+code[loc[1]:], lit, fmt.Sprint(n+1))
		if n > 0 {
			add(code[:loc[0]]+fmt.Sprint(n-1)+code[loc[1]:], lit, fmt.Sprint(n-1))
		}
	}

	// range(X) → range(X + 1) and range(A, B) → range(A, B + 1): the classic
	// inclusive-bound fixes.
	for _, loc := range reMutRange.FindAllStringSubmatchIndex(code, -1) {
		if inStringOrComment(code, loc[0]) {
			continue
		}
		arg := code[loc[2]:loc[3]]
		add(code[:loc[2]]+arg+" + 1"+code[loc[3]:], "range("+arg+")", "range("+arg+" + 1)")
	}
	for _, loc := range reMutRange2.FindAllStringSubmatchIndex(code, -1) {
		if inStringOrComment(code, loc[0]) {
			continue
		}
		a, b := code[loc[2]:loc[3]], code[loc[4]:loc[5]]
		add(code[:loc[4]]+b+" + 1"+code[loc[5]:],
			"range("+a+","+b+")", "range("+a+","+b+" + 1)")
	}
	return out
}

// inStringOrComment is a line-local heuristic: a position is skipped when an
// odd number of quotes precedes it on its line or a '#' starts earlier.
func inStringOrComment(code string, pos int) bool {
	lineStart := strings.LastIndexByte(code[:pos], '\n') + 1
	line := code[lineStart:pos]
	if strings.Contains(line, "#") {
		return true
	}
	return strings.Count(line, "'")%2 == 1 || strings.Count(line, `"`)%2 == 1
}

// runPasses reports whether code survives all asserts with a clean exit.
func runPasses(ctx context.Context, code string, asserts []string) bool {
	res, err := RunPython(ctx, code+"\n\n"+strings.Join(asserts, "\n"), mutRunTimeout)
	return err == nil && !res.TimedOut && res.ExitCode == 0
}
