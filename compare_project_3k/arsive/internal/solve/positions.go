package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// Positional race puzzles ("finished first", "exactly two places ahead of",
// "immediately behind") by permutation brute force — the shapes the pure
// transitive-ordering solver deliberately defers on (it cannot represent
// offsets or adjacency). n participants → n! candidate orders; n is small in
// puzzle prose, so plain code enumerates them all for zero tokens. Strictly
// self-gating: every sentence naming a participant must parse into exactly
// one constraint and the queried result must be unique, else defer.

type posCon func(pos map[string]int, n int) bool

var (
	rePosQuestion = regexp.MustCompile(`(?i)\b(?:give|what(?: is|'s)?|list|determine|state)\b.{0,40}\b(?:finishing\s+)?order\b|\bwho\s+(?:won|finished\s+(?:first|last|\w+))\b|\bfull\s+(?:finishing\s+)?order\b`)

	rePosFirst = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) first$|^([A-Z][a-z]*) won$`)
	rePosLast  = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) last$`)
	rePosNth   = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) (?:in )?(\w+)(?: place)?$`)
	rePosExact = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) exactly (\w+) places? (ahead of|in front of|before|behind|after) ([A-Z][a-z]*)$`)
	rePosAdj   = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) (?:immediately|right|directly|just) (ahead of|in front of|before|behind|after) ([A-Z][a-z]*)$`)
	rePosRel   = regexp.MustCompile(`^([A-Z][a-z]*) (?:finished|came(?: in)?|placed|was|is) (?:somewhere )?(ahead of|in front of|before|behind|after) ([A-Z][a-z]*)$`)

	rePosName = regexp.MustCompile(`[A-Z][a-z]*`)

	// A sentence must parse as a constraint only when it carries positional
	// semantics; name-enumeration intros ("Paco, Quin, Rita, and Sam raced")
	// are skipped rather than gating the solver off.
	rePosKeyword = regexp.MustCompile(`(?i)\b(?:finished?|came|placed?|places?|won|ahead|behind|before|after|first|last|second|third|fourth|fifth|sixth|seventh|eighth)\b`)

	posOrdinal = map[string]int{
		"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
		"sixth": 6, "seventh": 7, "eighth": 8, "1st": 1, "2nd": 2, "3rd": 3,
		"4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8,
	}
	posCount = map[string]int{
		"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
		"six": 6, "seven": 7, "eight": 8, "1": 1, "2": 2, "3": 3, "4": 4,
		"5": 5, "6": 6, "7": 7, "8": 8,
	}
)

// SolvePositions answers a positional-ordering puzzle, or defers.
func SolvePositions(prompt string) (string, bool) {
	q := rePosQuestion.FindString(prompt)
	if q == "" {
		return "", false
	}

	// Participants: capitalized tokens outside the stoplist. Ordinal words
	// appear lowercase mid-sentence; sentence-initial ones are stoplisted.
	var names []string
	seen := map[string]bool{}
	for _, w := range rePosName.FindAllString(prompt, -1) {
		lw := strings.ToLower(w)
		if !isName(w) || seen[w] || posOrdinal[lw] != 0 || posCount[lw] != 0 {
			continue
		}
		seen[w] = true
		names = append(names, w)
	}
	if len(names) < 3 || len(names) > 8 {
		return "", false
	}
	n := len(names)

	// Parse constraints sentence by sentence. Split on ':' too so an intro
	// like "In a five-person race with positions 1 through 5:" detaches from
	// the first clue. Sentences naming no participant are ignored; sentences
	// naming one must parse or we defer.
	var cons []posCon
	for _, raw := range strings.FieldsFunc(prompt, func(r rune) bool {
		return r == '.' || r == '?' || r == '!' || r == ':' || r == ';'
	}) {
		s := strings.TrimSpace(raw)
		if s == "" || rePosQuestion.MatchString(s) {
			continue
		}
		mentions := false
		for _, w := range rePosName.FindAllString(s, -1) {
			if seen[w] {
				mentions = true
				break
			}
		}
		if !mentions || !rePosKeyword.MatchString(s) {
			continue
		}
		c, ok := parsePosSentence(s, seen)
		if !ok {
			return "", false // a clue we cannot represent → defer
		}
		cons = append(cons, c)
	}
	if len(cons) < 2 {
		return "", false
	}

	// Brute force every permutation; collect orders consistent with all clues.
	var solutions [][]string
	for _, perm := range permutationsN(n) {
		pos := make(map[string]int, n)
		order := make([]string, n)
		for i, idx := range perm {
			pos[names[idx]] = i + 1
			order[i] = names[idx]
		}
		ok := true
		for _, c := range cons {
			if !c(pos, n) {
				ok = false
				break
			}
		}
		if ok {
			solutions = append(solutions, order)
			if len(solutions) > 24 {
				return "", false // hopelessly underdetermined
			}
		}
	}
	if len(solutions) == 0 {
		return "", false
	}
	return renderPosAnswer(q, solutions)
}

func parsePosSentence(s string, isPart map[string]bool) (posCon, bool) {
	name := func(w string) bool { return isPart[w] }
	if m := rePosFirst.FindStringSubmatch(s); m != nil {
		p := m[1] + m[2] // one group matches, the other is empty
		if !name(p) {
			return nil, false
		}
		return func(pos map[string]int, n int) bool { return pos[p] == 1 }, true
	}
	if m := rePosLast.FindStringSubmatch(s); m != nil {
		p := m[1]
		if !name(p) {
			return nil, false
		}
		return func(pos map[string]int, n int) bool { return pos[p] == n }, true
	}
	if m := rePosExact.FindStringSubmatch(s); m != nil {
		p1, k, rel, p2 := m[1], posCount[strings.ToLower(m[2])], m[3], m[4]
		if !name(p1) || !name(p2) || k == 0 {
			return nil, false
		}
		if rel == "behind" || rel == "after" {
			k = -k
		}
		// "p1 k places ahead of p2" → pos(p1) = pos(p2) - k
		return func(pos map[string]int, n int) bool { return pos[p1] == pos[p2]-k }, true
	}
	if m := rePosAdj.FindStringSubmatch(s); m != nil {
		p1, rel, p2 := m[1], m[2], m[3]
		if !name(p1) || !name(p2) {
			return nil, false
		}
		k := 1
		if rel == "behind" || rel == "after" {
			k = -1
		}
		return func(pos map[string]int, n int) bool { return pos[p1] == pos[p2]-k }, true
	}
	if m := rePosRel.FindStringSubmatch(s); m != nil {
		p1, rel, p2 := m[1], m[2], m[3]
		if !name(p1) || !name(p2) {
			return nil, false
		}
		if rel == "behind" || rel == "after" {
			return func(pos map[string]int, n int) bool { return pos[p1] > pos[p2] }, true
		}
		return func(pos map[string]int, n int) bool { return pos[p1] < pos[p2] }, true
	}
	if m := rePosNth.FindStringSubmatch(s); m != nil {
		p, k := m[1], posOrdinal[strings.ToLower(m[2])]
		if !name(p) || k == 0 {
			return nil, false
		}
		return func(pos map[string]int, n int) bool { return pos[p] == k }, true
	}
	return nil, false
}

// renderPosAnswer answers the question iff it is uniquely determined across
// every consistent order (the full order needs a single solution; "who won"
// only needs all solutions to agree on position 1).
func renderPosAnswer(q string, solutions [][]string) (string, bool) {
	lq := strings.ToLower(q)
	agreeAt := func(i int) (string, bool) {
		w := solutions[0][i]
		for _, s := range solutions[1:] {
			if s[i] != w {
				return "", false
			}
		}
		return w, true
	}
	switch {
	case strings.Contains(lq, "order"):
		if len(solutions) != 1 {
			return "", false
		}
		return "The order from first to last is " + strings.Join(solutions[0], ", ") + ".", true
	case strings.Contains(lq, "won") || strings.Contains(lq, "first"):
		if w, ok := agreeAt(0); ok {
			return fmt.Sprintf("%s finished first.", w), true
		}
	case strings.Contains(lq, "last"):
		if w, ok := agreeAt(len(solutions[0]) - 1); ok {
			return fmt.Sprintf("%s finished last.", w), true
		}
	default:
		// "who finished third" — extract the ordinal.
		for word, k := range posOrdinal {
			if strings.Contains(lq, word) && k <= len(solutions[0]) {
				if w, ok := agreeAt(k - 1); ok {
					return fmt.Sprintf("%s finished %s.", w, word), true
				}
			}
		}
	}
	return "", false
}

// permutationsN mirrors permutations() with the wider bound the race solver
// needs (n ≤ 8 → at most 40320 orders — still microseconds).
func permutationsN(n int) [][]int {
	if n < 1 || n > 8 {
		return nil
	}
	base := make([]int, n)
	for i := range base {
		base[i] = i
	}
	var out [][]int
	var rec func(k int)
	rec = func(k int) {
		if k == n {
			cp := make([]int, n)
			copy(cp, base)
			out = append(out, cp)
			return
		}
		for i := k; i < n; i++ {
			base[k], base[i] = base[i], base[k]
			rec(k + 1)
			base[k], base[i] = base[i], base[k]
		}
	}
	rec(0)
	return out
}
