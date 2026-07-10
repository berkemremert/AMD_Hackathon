package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// Single-attribute assignment puzzles — the shape of the guide's practice-07:
// "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird.
// Sam does not own the bird. Jo owns the dog. Who owns the cat?"
// One value domain (declared as a colon list or a single parenthetical),
// n people, owns/does-not-own clues, exhaustive n! assignment. Shares the
// zebra solver's clue grammar, permutation engine, self-gating rules, and
// query-cell-uniqueness answering.

var (
	// "... a different pet: cat, dog, bird." — a colon followed by a comma
	// list of ≥2 lowercase words, terminated by the sentence end.
	reSingleColonList = regexp.MustCompile(`:\s*([a-z]+(?:\s*,\s*[a-z]+)+(?:\s*,?\s*(?:and|or)\s+[a-z]+)?)\s*(?:[.?!]|$)`)
)

// SolveSingleAssign answers a one-domain assignment puzzle, or defers.
func SolveSingleAssign(prompt string) (string, bool) {
	values := singleDomain(prompt)
	if len(values) < 2 || len(values) > 6 {
		return "", false
	}
	valSet := map[string]bool{}
	for _, v := range values {
		if valSet[v] {
			return "", false // duplicate value → malformed
		}
		valSet[v] = true
	}

	// People: capitalized tokens that are neither stopwords nor values.
	var people []string
	seen := map[string]bool{}
	for _, w := range reZebraName.FindAllString(prompt, -1) {
		if !isName(w) || seen[w] || valSet[strings.ToLower(w)] {
			continue
		}
		seen[w] = true
		people = append(people, w)
	}
	if len(people) != len(values) {
		return "", false
	}
	personOf := map[string]string{}
	for _, p := range people {
		personOf[strings.ToLower(p)] = p
	}

	// Clues and queries, sentence by sentence. The declaration sentence is
	// exempt; any other owns/has sentence must parse or the solver defers.
	type con struct {
		person string
		val    string
		neg    bool
	}
	var cons []con
	var queries []zebraQuery
	for _, raw := range strings.FieldsFunc(prompt, func(r rune) bool { return r == '.' || r == '?' || r == '!' }) {
		s := strings.TrimSpace(raw)
		if s == "" || !reZebraKeyword.MatchString(s) {
			continue
		}
		// The declaration sentence carries the value list, not a clue.
		if strings.Contains(s, "(") || reSingleColonList.MatchString(s) {
			continue
		}
		switch {
		case reZebraQWhoOwns.MatchString(s):
			m := reZebraQWhoOwns.FindStringSubmatch(s)
			v := strings.ToLower(m[1])
			if !valSet[v] {
				return "", false
			}
			queries = append(queries, zebraQuery{kind: "whoHasVal", val: v})
		case reZebraQOwnWhat.MatchString(s):
			m := reZebraQOwnWhat.FindStringSubmatch(s)
			p, ok := personOf[strings.ToLower(m[1])]
			if !ok {
				return "", false
			}
			queries = append(queries, zebraQuery{kind: "valOfPerson", person: p})
		case reZebraOwnsNeither.MatchString(s):
			m := reZebraOwnsNeither.FindStringSubmatch(s)
			p, okP := personOf[strings.ToLower(m[1])]
			v1, v2 := strings.ToLower(m[2]), strings.ToLower(m[3])
			if !okP || !valSet[v1] || !valSet[v2] {
				return "", false
			}
			cons = append(cons, con{p, v1, true}, con{p, v2, true})
		case reZebraNotOwn.MatchString(s):
			m := reZebraNotOwn.FindStringSubmatch(s)
			p, okP := personOf[strings.ToLower(m[1])]
			v := strings.ToLower(m[2])
			if !okP || !valSet[v] {
				return "", false
			}
			cons = append(cons, con{p, v, true})
		case reZebraOwns.MatchString(s):
			m := reZebraOwns.FindStringSubmatch(s)
			p, okP := personOf[strings.ToLower(m[1])]
			v := strings.ToLower(m[2])
			if !okP || !valSet[v] {
				return "", false
			}
			cons = append(cons, con{p, v, false})
		default:
			return "", false // an owns/has clue we cannot represent → defer
		}
	}
	if len(cons) < 1 || len(queries) == 0 {
		return "", false
	}

	// Exhaust all n! assignments; the queried cells must agree across every
	// consistent one (the full grid may stay ambiguous).
	var answers []string
	found := false
	for _, perm := range permutations(len(people)) {
		valOf := map[string]string{}
		personWith := map[string]string{}
		for i, p := range people {
			valOf[p] = values[perm[i]]
			personWith[values[perm[i]]] = p
		}
		ok := true
		for _, c := range cons {
			if (valOf[c.person] == c.val) == c.neg {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		found = true
		vals := make([]string, len(queries))
		for qi, q := range queries {
			if q.kind == "whoHasVal" {
				vals[qi] = fmt.Sprintf("%s owns the %s", personWith[q.val], q.val)
			} else {
				vals[qi] = fmt.Sprintf("%s owns the %s", q.person, valOf[q.person])
			}
		}
		if answers == nil {
			answers = vals
			continue
		}
		for i := range vals {
			if vals[i] != answers[i] {
				return "", false // queried cell not uniquely determined
			}
		}
	}
	if !found || answers == nil {
		return "", false
	}
	return strings.Join(answers, ", and ") + ".", true
}

// singleDomain finds exactly one value list: a colon list, or a single
// parenthetical (two parentheticals belong to the zebra solver).
func singleDomain(prompt string) []string {
	parens := reZebraDomain.FindAllStringSubmatchIndex(prompt, -1)
	colon := reSingleColonList.FindStringSubmatchIndex(prompt)
	var inner []int
	switch {
	case colon != nil && len(parens) == 0:
		inner = colon[2:4]
	case colon == nil && len(parens) == 1:
		inner = parens[0][2:4]
	default:
		return nil
	}
	var values []string
	for _, v := range regexp.MustCompile(`[a-z]+`).FindAllString(prompt[inner[0]:inner[1]], -1) {
		if v != "and" && v != "or" {
			values = append(values, v)
		}
	}
	return values
}
