package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// Knights-and-knaves by brute force: with n speakers there are only 2^n
// truth assignments, so plain code tries them all — zero tokens. The hard
// part is parsing English statements into predicates, so the solver is
// strictly self-gating: any quoted statement it cannot parse means defer.
// A wrong answer breaks the accuracy gate; a deferral only costs tokens.

type kkStmt struct {
	speaker string
	eval    func(assign map[string]bool) bool // true under this assignment?
}

var (
	reKKGate = regexp.MustCompile(`(?i)\bknights?\b[\s\S]*\bknaves?\b|\bknaves?\b[\s\S]*\bknights?\b`)
	// "A says 'B is a knave.'" — straight or curly quotes, optional colon.
	reKKSays = regexp.MustCompile(`([A-Z][a-z]*)\s+(?:says?|said|claims?|state[sd]?)[,:]?\s*['‘’"“”]([^'‘’"“”]+)['‘’"“”]`)
	// Every says/claims occurrence must be captured by reKKSays, else defer.
	reKKSaysAny = regexp.MustCompile(`(?i)\b(?:says?|said|claims?|state[sd]?)\b`)

	reKKIsType    = regexp.MustCompile(`(?i)^([A-Z][a-z]*) is a (knight|knave)$`)
	reKKBoth      = regexp.MustCompile(`(?i)^(?:both )?([A-Z][a-z]*) and ([A-Z][a-z]*) are (?:both )?(knights|knaves)$`)
	reKKSame      = regexp.MustCompile(`(?i)^([A-Z][a-z]*) and ([A-Z][a-z]*) are (?:of )?the same(?: type)?$`)
	reKKDiff      = regexp.MustCompile(`(?i)^([A-Z][a-z]*) and ([A-Z][a-z]*) are (?:of )?different(?: types?)?$`)
	reKKIAm       = regexp.MustCompile(`(?i)^I am a (knight|knave)$`)
	reKKWe        = regexp.MustCompile(`(?i)^we are (?:both|all) (knights|knaves)$`)
	// The question often names people who never speak ("What are A and B?");
	// they are still part of "we", so harvest them from the enumeration.
	reKKWhoList = regexp.MustCompile(`(?:[Ww]hat (?:are|is)|each of)\s+((?:[A-Z][a-z]*)(?:(?:,\s*|,?\s+and\s+|\s+or\s+)[A-Z][a-z]*)*)`)
	reKKAtLeast   = regexp.MustCompile(`(?i)^at least one of ([A-Z][a-z]*) and ([A-Z][a-z]*) is a (knight|knave)$`)
	reKKExactlyOne = regexp.MustCompile(`(?i)^exactly one of ([A-Z][a-z]*) and ([A-Z][a-z]*) is a (knight|knave)$`)
)

// SolveKnights answers a knights-and-knaves puzzle when every statement
// parses and exactly one truth assignment is consistent.
func SolveKnights(prompt string) (string, bool) {
	if !reKKGate.MatchString(prompt) {
		return "", false
	}
	matches := reKKSays.FindAllStringSubmatch(prompt, -1)
	if len(matches) < 1 {
		return "", false // single-statement puzzles are fine; uniqueness gates below
	}
	// Self-gate: an unquoted or unmatched "says" means a statement we did not
	// capture — the model must handle it.
	if len(reKKSaysAny.FindAllString(prompt, -1)) != len(matches) {
		return "", false
	}

	var persons []string // insertion-ordered
	seen := map[string]bool{}
	addPerson := func(name string) {
		if name != "" && !seen[name] {
			seen[name] = true
			persons = append(persons, name)
		}
	}

	var stmts []kkStmt
	for _, m := range matches {
		speaker, text := m[1], strings.TrimSpace(strings.TrimRight(strings.TrimSpace(m[2]), ".!"))
		if !isName(speaker) {
			return "", false
		}
		addPerson(speaker)
		eval, refs, ok := parseKKStatement(speaker, text)
		if !ok {
			return "", false // unparsed statement → defer
		}
		for _, r := range refs {
			addPerson(r)
		}
		stmts = append(stmts, kkStmt{speaker: speaker, eval: eval})
	}
	// Silent participants named in the question ("What are A and B?").
	for _, m := range reKKWhoList.FindAllStringSubmatch(prompt, -1) {
		for _, w := range regexp.MustCompile(`[A-Z][a-z]*`).FindAllString(m[1], -1) {
			if isName(w) {
				addPerson(w)
			}
		}
	}
	if len(persons) < 2 || len(persons) > 12 {
		return "", false
	}

	// Brute force all 2^n assignments (true = knight).
	var solutions []map[string]bool
	for mask := 0; mask < 1<<len(persons); mask++ {
		assign := make(map[string]bool, len(persons))
		for i, p := range persons {
			assign[p] = mask&(1<<i) != 0
		}
		ok := true
		for _, s := range stmts {
			// A knight's statement is true, a knave's is false.
			if s.eval(assign) != assign[s.speaker] {
				ok = false
				break
			}
		}
		if ok {
			solutions = append(solutions, assign)
			if len(solutions) > 1 {
				return "", false // ambiguous → defer
			}
		}
	}
	if len(solutions) != 1 {
		return "", false // contradictory premises → defer
	}

	sol := solutions[0]
	parts := make([]string, len(persons))
	for i, p := range persons {
		kind := "knave"
		if sol[p] {
			kind = "knight"
		}
		parts[i] = fmt.Sprintf("%s is a %s", p, kind)
	}
	return joinAnd(parts) + ".", true
}

// parseKKStatement compiles one quoted statement into an evaluator plus the
// names it references. ok=false on any phrasing outside the model.
func parseKKStatement(speaker, text string) (func(map[string]bool) bool, []string, bool) {
	if m := reKKIsType.FindStringSubmatch(text); m != nil {
		name, wantKnight := m[1], strings.EqualFold(m[2], "knight")
		if !isName(name) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return a[name] == wantKnight }, []string{name}, true
	}
	if m := reKKBoth.FindStringSubmatch(text); m != nil {
		x, y, wantKnight := m[1], m[2], strings.EqualFold(m[3], "knights")
		if !isName(x) || !isName(y) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return a[x] == wantKnight && a[y] == wantKnight }, []string{x, y}, true
	}
	if m := reKKSame.FindStringSubmatch(text); m != nil {
		x, y := m[1], m[2]
		if !isName(x) || !isName(y) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return a[x] == a[y] }, []string{x, y}, true
	}
	if m := reKKDiff.FindStringSubmatch(text); m != nil {
		x, y := m[1], m[2]
		if !isName(x) || !isName(y) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return a[x] != a[y] }, []string{x, y}, true
	}
	if m := reKKIAm.FindStringSubmatch(text); m != nil {
		wantKnight := strings.EqualFold(m[1], "knight")
		return func(a map[string]bool) bool { return a[speaker] == wantKnight }, nil, true
	}
	if m := reKKWe.FindStringSubmatch(text); m != nil {
		// "We are both/all X" quantifies over every participant.
		wantKnight := strings.EqualFold(m[1], "knights")
		return func(a map[string]bool) bool {
			for _, v := range a {
				if v != wantKnight {
					return false
				}
			}
			return true
		}, nil, true
	}
	if m := reKKAtLeast.FindStringSubmatch(text); m != nil {
		x, y, wantKnight := m[1], m[2], strings.EqualFold(m[3], "knight")
		if !isName(x) || !isName(y) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return a[x] == wantKnight || a[y] == wantKnight }, []string{x, y}, true
	}
	if m := reKKExactlyOne.FindStringSubmatch(text); m != nil {
		x, y, wantKnight := m[1], m[2], strings.EqualFold(m[3], "knight")
		if !isName(x) || !isName(y) {
			return nil, nil, false
		}
		return func(a map[string]bool) bool { return (a[x] == wantKnight) != (a[y] == wantKnight) }, []string{x, y}, true
	}
	return nil, nil, false
}

// joinAnd renders "a", "a and b", or "a, b, and c".
func joinAnd(parts []string) string {
	switch len(parts) {
	case 0:
		return ""
	case 1:
		return parts[0]
	case 2:
		return parts[0] + " and " + parts[1]
	default:
		return strings.Join(parts[:len(parts)-1], ", ") + ", and " + parts[len(parts)-1]
	}
}
