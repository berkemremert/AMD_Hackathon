package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// Zebra-style attribute grids ("four friends, each owns one pet and lives in
// a colored house") by exhaustive assignment: with n people and two attribute
// domains there are only (n!)² combinations — trivial for plain code. The
// solver is strictly self-gating: the puzzle must declare exactly two value
// domains in parentheses, every clue sentence must parse, and every query
// must be uniquely determined across ALL consistent assignments (so it can
// answer even when the full grid is underdetermined but the asked cells are
// not). Anything else → defer to the model.

type zebraCon struct {
	person string // "" for owner-link constraints
	dom    int    // domain index of val
	val    string
	neg    bool   // person does NOT have val
	link   string // value from the other domain held by the same person
}

type zebraQuery struct {
	kind   string // "whoHasVal" | "valOfPerson"
	person string
	dom    int  // concrete domain index (resolved before solving)
	val    string
	house  bool // render with house phrasing
}

type zebraParseResult struct {
	cons    []zebraCon
	queries []zebraQuery
	ownDom  int // domain used with owns/has (-1 if never seen)
	houseDom int // domain used with "the X house" (-1 if never seen)
}

var (
	// A domain list: "(cat, dog, fish, bird)" — 3+ lowercase words.
	reZebraDomain = regexp.MustCompile(`\(([a-z]+(?:\s*,\s*[a-z]+)+(?:\s*,?\s*(?:and|or)\s+[a-z]+)?)\)`)

	reZebraOwnsNeither  = regexp.MustCompile(`(?i)^(\w+) (?:owns|has) neither (?:the |a |an )?(\w+) nor (?:the |a |an )?(\w+)$`)
	reZebraNotOwn       = regexp.MustCompile(`(?i)^(\w+) does not (?:own|have) (?:the |a |an )?(\w+)$`)
	reZebraOwns         = regexp.MustCompile(`(?i)^(\w+) (?:owns|has) (?:the |a |an )?(\w+)$`)
	reZebraOwnerLives   = regexp.MustCompile(`(?i)^the (\w+) owner lives in the (\w+) house$`)
	reZebraHouseOwns    = regexp.MustCompile(`(?i)^the person (?:living |who lives )?in the (\w+) house (?:owns|has) (?:the |a |an )?(\w+)$`)
	reZebraLivesNeither = regexp.MustCompile(`(?i)^(\w+) lives in neither the (\w+) nor the (\w+)(?: house)?$`)
	reZebraNotLive      = regexp.MustCompile(`(?i)^(\w+) does not live in the (\w+)(?: house)?$`)
	reZebraLives        = regexp.MustCompile(`(?i)^(\w+) lives in the (\w+)(?: house)?$`)

	reZebraQWhoOwns  = regexp.MustCompile(`(?i)who (?:owns|has) (?:the |a |an )?(\w+)`)
	reZebraQColor    = regexp.MustCompile(`(?i)(?:what|which) colou?r is (\w+)(?:'s)? house`)
	reZebraQWhoLives = regexp.MustCompile(`(?i)who lives in the (\w+)(?: house)?`)
	reZebraQOwnWhat  = regexp.MustCompile(`(?i)(?:what|which) (?:pet |animal )?does (\w+) (?:own|have)`)

	reZebraName = regexp.MustCompile(`[A-Z][a-z]+`)

	// Only sentences with assignment semantics must parse; enumeration intros
	// without a constraint verb are skipped rather than gating the solver off.
	reZebraKeyword = regexp.MustCompile(`(?i)\b(?:owns?|has|have|lives?|house|owner)\b`)
)

// SolveZebra answers a two-domain assignment puzzle, or defers.
func SolveZebra(prompt string) (string, bool) {
	// Exactly two declared domains of equal size.
	domMatches := reZebraDomain.FindAllStringSubmatch(prompt, -1)
	if len(domMatches) != 2 {
		return "", false
	}
	domains := make([][]string, 2)
	valDom := map[string]int{} // value → domain index
	for i, m := range domMatches {
		for _, v := range regexp.MustCompile(`[a-z]+`).FindAllString(m[1], -1) {
			if v == "and" || v == "or" {
				continue
			}
			if _, dup := valDom[v]; dup {
				return "", false // same value in both domains → ambiguous
			}
			valDom[v] = i
			domains[i] = append(domains[i], v)
		}
	}
	if len(domains[0]) != len(domains[1]) || len(domains[0]) < 3 || len(domains[0]) > 6 {
		return "", false
	}
	n := len(domains[0])

	// People: capitalized tokens that are neither stopwords nor domain values.
	var people []string
	seen := map[string]bool{}
	for _, w := range reZebraName.FindAllString(prompt, -1) {
		if !isName(w) || seen[w] {
			continue
		}
		if _, isVal := valDom[strings.ToLower(w)]; isVal {
			continue
		}
		seen[w] = true
		people = append(people, w)
	}
	if len(people) != n {
		return "", false
	}
	personOf := map[string]string{} // lowercase → canonical
	for _, p := range people {
		personOf[strings.ToLower(p)] = p
	}

	pr, ok := zebraParse(prompt, personOf, valDom)
	if !ok || len(pr.cons) < 2 || len(pr.queries) == 0 {
		return "", false
	}
	// Resolve role-relative query domains to concrete indices and cross-check
	// value-based queries against the detected roles.
	for i := range pr.queries {
		q := &pr.queries[i]
		if q.kind == "valOfPerson" {
			want := pr.ownDom
			if q.house {
				want = pr.houseDom
			}
			if want == -1 {
				return "", false // role never established by any clue
			}
			q.dom = want
		} else if q.house && pr.houseDom != -1 && q.dom != pr.houseDom {
			return "", false
		} else if !q.house && pr.ownDom != -1 && q.dom != pr.ownDom {
			return "", false
		}
	}

	// Enumerate assignments: permutations of each domain over people.
	perms := permutations(n)
	if perms == nil {
		return "", false
	}
	var answers []string // rendered answer per query, must agree across solutions
	solutionCount := 0
	for _, p0 := range perms {
		for _, p1 := range perms {
			// assignment: people[i] holds domains[0][p0[i]] and domains[1][p1[i]]
			has := func(personIdx, dom int) string {
				if dom == 0 {
					return domains[0][p0[personIdx]]
				}
				return domains[1][p1[personIdx]]
			}
			if !zebraConsistent(pr.cons, people, has) {
				continue
			}
			solutionCount++
			vals := zebraAnswer(pr.queries, people, has)
			if vals == nil {
				return "", false
			}
			if answers == nil {
				answers = vals
			} else {
				for i := range vals {
					if vals[i] != answers[i] {
						return "", false // queried cell not uniquely determined
					}
				}
			}
		}
	}
	if solutionCount == 0 || answers == nil {
		return "", false
	}
	return strings.Join(answers, ", and ") + ".", true
}

func zebraParse(prompt string, personOf map[string]string, valDom map[string]int) (zebraParseResult, bool) {
	pr := zebraParseResult{ownDom: -1, houseDom: -1}
	fail := func() (zebraParseResult, bool) { return pr, false }

	person := func(s string) (string, bool) { p, ok := personOf[strings.ToLower(s)]; return p, ok }
	val := func(s string) (string, int, bool) {
		v := strings.ToLower(s)
		d, ok := valDom[v]
		return v, d, ok
	}
	// The first owns/house clue fixes which parenthetical plays which role;
	// later clues must agree.
	fixOwn := func(d int) bool {
		if pr.ownDom == -1 {
			pr.ownDom = d
			if pr.houseDom == d {
				return false
			}
			return true
		}
		return pr.ownDom == d
	}
	fixHouse := func(d int) bool {
		if pr.houseDom == -1 {
			pr.houseDom = d
			if pr.ownDom == d {
				return false
			}
			return true
		}
		return pr.houseDom == d
	}

	for _, raw := range strings.FieldsFunc(prompt, func(r rune) bool { return r == '.' || r == '?' || r == '!' }) {
		s := strings.TrimSpace(raw)
		if s == "" {
			continue
		}
		if strings.Contains(s, "(") {
			continue // intro sentence carrying the domain lists
		}
		if !reZebraKeyword.MatchString(s) {
			continue // no assignment semantics (e.g. a name-list intro)
		}
		if q, ok := zebraParseQuery(s, personOf, valDom); ok {
			pr.queries = append(pr.queries, q...)
			continue
		}
		switch {
		case reZebraOwnsNeither.MatchString(s):
			m := reZebraOwnsNeither.FindStringSubmatch(s)
			p, okP := person(m[1])
			v1, d1, ok1 := val(m[2])
			v2, d2, ok2 := val(m[3])
			if !okP || !ok1 || !ok2 || d1 != d2 || !fixOwn(d1) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d1, val: v1, neg: true},
				zebraCon{person: p, dom: d2, val: v2, neg: true})
		case reZebraNotOwn.MatchString(s):
			m := reZebraNotOwn.FindStringSubmatch(s)
			p, okP := person(m[1])
			v, d, okV := val(m[2])
			if !okP || !okV || !fixOwn(d) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d, val: v, neg: true})
		case reZebraOwnerLives.MatchString(s):
			m := reZebraOwnerLives.FindStringSubmatch(s)
			v1, d1, ok1 := val(m[1]) // owned value
			v2, d2, ok2 := val(m[2]) // house value
			if !ok1 || !ok2 || d1 == d2 || !fixOwn(d1) || !fixHouse(d2) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{dom: d1, val: v1, link: v2})
		case reZebraHouseOwns.MatchString(s):
			m := reZebraHouseOwns.FindStringSubmatch(s)
			v1, d1, ok1 := val(m[1]) // house value
			v2, d2, ok2 := val(m[2]) // owned value
			if !ok1 || !ok2 || d1 == d2 || !fixHouse(d1) || !fixOwn(d2) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{dom: d2, val: v2, link: v1})
		case reZebraLivesNeither.MatchString(s):
			m := reZebraLivesNeither.FindStringSubmatch(s)
			p, okP := person(m[1])
			v1, d1, ok1 := val(m[2])
			v2, d2, ok2 := val(m[3])
			if !okP || !ok1 || !ok2 || d1 != d2 || !fixHouse(d1) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d1, val: v1, neg: true},
				zebraCon{person: p, dom: d2, val: v2, neg: true})
		case reZebraNotLive.MatchString(s):
			m := reZebraNotLive.FindStringSubmatch(s)
			p, okP := person(m[1])
			v, d, okV := val(m[2])
			if !okP || !okV || !fixHouse(d) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d, val: v, neg: true})
		case reZebraLives.MatchString(s):
			m := reZebraLives.FindStringSubmatch(s)
			p, okP := person(m[1])
			v, d, okV := val(m[2])
			if !okP || !okV || !fixHouse(d) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d, val: v})
		case reZebraOwns.MatchString(s):
			m := reZebraOwns.FindStringSubmatch(s)
			p, okP := person(m[1])
			v, d, okV := val(m[2])
			if !okP || !okV || !fixOwn(d) {
				return fail()
			}
			pr.cons = append(pr.cons, zebraCon{person: p, dom: d, val: v})
		default:
			return fail() // unparsed clue → defer
		}
	}
	return pr, true
}

func zebraParseQuery(s string, personOf map[string]string, valDom map[string]int) ([]zebraQuery, bool) {
	var out []zebraQuery
	for _, part := range regexp.MustCompile(`(?i),?\s*\band\b`).Split(s, -1) {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		switch {
		case reZebraQWhoOwns.MatchString(part):
			m := reZebraQWhoOwns.FindStringSubmatch(part)
			v := strings.ToLower(m[1])
			d, ok := valDom[v]
			if !ok {
				return nil, false
			}
			out = append(out, zebraQuery{kind: "whoHasVal", dom: d, val: v})
		case reZebraQWhoLives.MatchString(part):
			m := reZebraQWhoLives.FindStringSubmatch(part)
			v := strings.ToLower(m[1])
			d, ok := valDom[v]
			if !ok {
				return nil, false
			}
			out = append(out, zebraQuery{kind: "whoHasVal", dom: d, val: v, house: true})
		case reZebraQColor.MatchString(part):
			m := reZebraQColor.FindStringSubmatch(part)
			p, ok := personOf[strings.ToLower(m[1])]
			if !ok {
				return nil, false
			}
			out = append(out, zebraQuery{kind: "valOfPerson", person: p, house: true})
		case reZebraQOwnWhat.MatchString(part):
			m := reZebraQOwnWhat.FindStringSubmatch(part)
			p, ok := personOf[strings.ToLower(m[1])]
			if !ok {
				return nil, false
			}
			out = append(out, zebraQuery{kind: "valOfPerson", person: p})
		default:
			return nil, false
		}
	}
	return out, len(out) > 0
}

func zebraConsistent(cons []zebraCon, people []string, has func(int, int) string) bool {
	idxOf := func(name string) int {
		for i, p := range people {
			if p == name {
				return i
			}
		}
		return -1
	}
	for _, c := range cons {
		if c.link != "" {
			// The person holding c.val (domain c.dom) also holds c.link.
			holder := -1
			for i := range people {
				if has(i, c.dom) == c.val {
					holder = i
					break
				}
			}
			if holder == -1 || has(holder, 1-c.dom) != c.link {
				return false
			}
			continue
		}
		i := idxOf(c.person)
		if i == -1 {
			return false
		}
		if (has(i, c.dom) == c.val) == c.neg {
			return false
		}
	}
	return true
}

// zebraAnswer renders one answer string per query under a given assignment.
func zebraAnswer(queries []zebraQuery, people []string, has func(int, int) string) []string {
	out := make([]string, len(queries))
	for qi, q := range queries {
		switch q.kind {
		case "whoHasVal":
			holder := ""
			for i, p := range people {
				if has(i, q.dom) == q.val {
					holder = p
					break
				}
			}
			if holder == "" {
				return nil
			}
			if q.house {
				out[qi] = fmt.Sprintf("%s lives in the %s house", holder, q.val)
			} else {
				out[qi] = fmt.Sprintf("%s owns the %s", holder, q.val)
			}
		case "valOfPerson":
			i := -1
			for j, p := range people {
				if p == q.person {
					i = j
					break
				}
			}
			if i == -1 {
				return nil
			}
			if q.house {
				out[qi] = fmt.Sprintf("%s's house is %s", q.person, has(i, q.dom))
			} else {
				out[qi] = fmt.Sprintf("%s owns the %s", q.person, has(i, q.dom))
			}
		}
	}
	return out
}

// permutations returns all permutations of 0..n-1 (n ≤ 6 → at most 720).
func permutations(n int) [][]int {
	if n < 1 || n > 6 {
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
