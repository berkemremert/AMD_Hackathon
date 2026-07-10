package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// Free logic solvers: translate the two most regular logic-puzzle shapes into
// plain graph problems and solve them in Go — zero API tokens. Both are
// deliberately conservative: they return ok=false on any ambiguity, negation,
// contradiction, or out-of-model phrasing, deferring to the model. A wrong
// answer breaks the accuracy gate (unrecoverable); an unneeded deferral costs
// only tokens (cheap).

// --- 1. Transitive ordering ("who finished first / last / the order") ---

var (
	reOrderQuestion = regexp.MustCompile(`(?i)\b(who\s+(won|finished\s+first|came\s+first|is\s+first|was\s+first|placed\s+first)|who\s+(finished\s+)?last|what\s+is\s+the\s+(finishing\s+)?order|full\s+(finishing\s+)?order|order\s+of\s+finish)\b`)

	reBeforeAfter = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+)?before\s+([A-Z][a-z]+)\s+(?:but\s+|and\s+)?after\s+([A-Z][a-z]+)`)
	reAfterBefore = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+)?after\s+([A-Z][a-z]+)\s+(?:but\s+|and\s+)?before\s+([A-Z][a-z]+)`)
	reBefore      = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+)?(?:before|ahead of)\s+([A-Z][a-z]+)`)
	reAfter       = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+)?(?:after|behind)\s+([A-Z][a-z]+)`)
	reLast        = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+|was\s+|is\s+)?last\b`)
	reFirstWon    = regexp.MustCompile(`([A-Z][a-z]+)\s+(?:finished\s+|came\s+in\s+|placed\s+|was\s+|is\s+)?(?:first|won)\b`)

	// Positional/adjacency phrasing our precedence model cannot represent.
	reAdjacency = regexp.MustCompile(`(?i)immediately|next to|(?:to the |on the )?(?:left|right)\b|exactly\s+\w+\s+(?:place|position)|two places|\bseat`)
	reNegation  = regexp.MustCompile(`(?i)\bnot\b|n't`)

	// Capitalized tokens that are never participant names in these puzzles.
	nameStop = map[string]bool{
		"Who": true, "What": true, "When": true, "Where": true, "Why": true,
		"How": true, "The": true, "And": true, "But": true, "If": true,
		"In": true, "On": true, "At": true, "Answer": true, "First": true,
		"Last": true, "Four": true, "Five": true, "Three": true, "Two": true,
		"Six": true, "Seven": true, "Eight": true, "Nine": true, "Ten": true,
		"Give": true, "So": true, "Then": true, "Each": true, "All": true,
	}
)

func isName(w string) bool { return w != "" && !nameStop[w] }

// SolveOrdering handles pure before/after/first/last race-ordering puzzles.
func SolveOrdering(prompt string) (string, bool) {
	q := reOrderQuestion.FindString(prompt)
	if q == "" {
		return "", false
	}
	// Negation or positional constraints are out of this model's scope.
	if reAdjacency.MatchString(prompt) || reNegation.MatchString(prompt) {
		return "", false
	}

	names := map[string]bool{}
	adj := map[string]map[string]bool{}
	addEdge := func(earlier, later string) {
		if !isName(earlier) || !isName(later) || earlier == later {
			return
		}
		names[earlier], names[later] = true, true
		if adj[earlier] == nil {
			adj[earlier] = map[string]bool{}
		}
		adj[earlier][later] = true
	}

	edges := 0
	// Compound clauses first (they subsume the plain before/after matches).
	for _, m := range reBeforeAfter.FindAllStringSubmatch(prompt, -1) {
		addEdge(m[1], m[2]) // m1 before m2
		addEdge(m[3], m[1]) // m1 after m3 => m3 before m1
		edges += 2
	}
	for _, m := range reAfterBefore.FindAllStringSubmatch(prompt, -1) {
		addEdge(m[2], m[1]) // m1 after m2 => m2 before m1
		addEdge(m[1], m[3]) // m1 before m3
		edges += 2
	}
	for _, m := range reBefore.FindAllStringSubmatch(prompt, -1) {
		addEdge(m[1], m[2])
		edges++
	}
	for _, m := range reAfter.FindAllStringSubmatch(prompt, -1) {
		addEdge(m[2], m[1])
		edges++
	}
	// last/first need the full participant set, so record them, then apply.
	var lastNames, firstNames []string
	for _, m := range reLast.FindAllStringSubmatch(prompt, -1) {
		if isName(m[1]) {
			names[m[1]] = true
			lastNames = append(lastNames, m[1])
		}
	}
	for _, m := range reFirstWon.FindAllStringSubmatch(prompt, -1) {
		if isName(m[1]) {
			names[m[1]] = true
			firstNames = append(firstNames, m[1])
		}
	}
	for _, l := range lastNames {
		for n := range names {
			if n != l {
				addEdge(n, l)
				edges++
			}
		}
	}
	for _, f := range firstNames {
		for n := range names {
			if n != f {
				addEdge(f, n)
				edges++
			}
		}
	}
	if edges < 2 || len(names) < 2 {
		return "", false
	}

	nameList := make([]string, 0, len(names))
	for n := range names {
		nameList = append(nameList, n)
	}
	order, ok := uniqueTopoOrder(nameList, adj)
	if !ok {
		return "", false // ambiguous, cyclic, or incomplete — defer
	}

	switch {
	case regexp.MustCompile(`(?i)who\s+(won|.*first)`).MatchString(q):
		return fmt.Sprintf("%s finished first.", order[0]), true
	case regexp.MustCompile(`(?i)last`).MatchString(q):
		return fmt.Sprintf("%s finished last.", order[len(order)-1]), true
	default:
		return "The order from first to last is " + strings.Join(order, ", ") + ".", true
	}
}

// uniqueTopoOrder returns the total order iff exactly one topological order
// covers every name (i.e. at every step exactly one node has in-degree 0).
func uniqueTopoOrder(names []string, adj map[string]map[string]bool) ([]string, bool) {
	indeg := map[string]int{}
	for _, n := range names {
		indeg[n] = 0
	}
	for u := range adj {
		for v := range adj[u] {
			indeg[v]++
		}
	}
	var order []string
	remaining := map[string]bool{}
	for _, n := range names {
		remaining[n] = true
	}
	for len(remaining) > 0 {
		var zero []string
		for n := range remaining {
			if indeg[n] == 0 {
				zero = append(zero, n)
			}
		}
		if len(zero) != 1 {
			return nil, false // 0 = cycle, >1 = ambiguous
		}
		n := zero[0]
		order = append(order, n)
		delete(remaining, n)
		for v := range adj[n] {
			indeg[v]--
		}
	}
	return order, true
}

// --- 2. Syllogism ("all X are Y; are all X Z?") ---

var (
	reUniversal = regexp.MustCompile(`(?i)\b(?:all|every|each)\s+([A-Za-z]+?)s?\s+(?:are|is)\s+(?:a\s+|an\s+)?([A-Za-z]+?)s?\b`)
	reSyllQ     = regexp.MustCompile(`(?i)\b(?:are|is)\s+(?:all|every|each)\s+([A-Za-z]+?)s?\s+(?:necessarily\s+|definitely\s+|also\s+)?(?:a\s+|an\s+)?([A-Za-z]+?)s?\b`)
	// Non-universal quantifier that actually quantifies a subject noun ("some
	// cats are"), NOT the bare "no" in "answer yes or no".
	reNonUniv = regexp.MustCompile(`(?i)\b(some|most|few|no|not all|neither)\s+[A-Za-z]+\s+(?:are|is)\b`)
)

// SolveSyllogism handles chained universal-affirmative syllogisms.
func SolveSyllogism(prompt string) (string, bool) {
	if reNonUniv.MatchString(prompt) {
		return "", false // non-universal premises need different logic
	}
	q := reSyllQ.FindStringSubmatch(prompt)
	if q == nil {
		return "", false
	}
	src, dst := normTerm(q[1]), normTerm(q[2])
	if src == "" || dst == "" || src == dst {
		return "", false
	}

	adj := map[string]map[string]bool{}
	prems := 0
	for _, m := range reUniversal.FindAllStringSubmatch(prompt, -1) {
		a, b := normTerm(m[1]), normTerm(m[2])
		if a == "" || b == "" || a == b {
			continue
		}
		if adj[a] == nil {
			adj[a] = map[string]bool{}
		}
		adj[a][b] = true
		prems++
	}
	if prems < 2 {
		return "", false
	}
	if reachable(src, dst, adj, map[string]bool{}) {
		return fmt.Sprintf("Yes — by transitivity, every %s is a %s.", titleFirst(src), titleFirst(dst)), true
	}
	// We only assert Yes with proof; a missing chain isn't a proof of No.
	return "", false
}

func normTerm(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	return strings.TrimSuffix(s, "s")
}

func titleFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func reachable(from, to string, adj map[string]map[string]bool, seen map[string]bool) bool {
	if from == to {
		return true
	}
	if seen[from] {
		return false
	}
	seen[from] = true
	for next := range adj[from] {
		if reachable(next, to, adj, seen) {
			return true
		}
	}
	return false
}
