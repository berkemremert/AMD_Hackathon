package solve

import (
	"fmt"
	"regexp"
	"strings"
)

// DeriveAsserts turns example input/output pairs stated in a task prompt into
// Python asserts — no model involved, so verification stays at zero tokens.
// Handles the common spec phrasings:
//
//	fib(10) returns 55            valid_brackets('([)]') is False
//	parse_duration('2h30m')==9000  reverse_words('a b') -> 'b a'
//	Example: 'hello world foo' becomes 'foo world hello'   (with a named function)
const (
	assertLit = `'[^']*'|"[^"]*"|-?\d[\d_,]*(?:\.\d+)?|True|False|None|\[[^\]]*\]|\([^)]*\)`
	assertOp  = `(?:==|should\s+(?:return|be|equal|give|yield)|must\s+(?:return|be|equal)|equals|returns?|is|->|→|becomes|gives|yields)`
	assertCall = `(\w+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*` + assertOp + `\s*`
)

var (
	reCallExample = regexp.MustCompile(assertCall + `(` + assertLit + `)`)
	// Debug-task phrasing: "f(5) returns 24 instead of 120" — the SECOND
	// literal is the expected value; the first is the observed bug. Matched
	// (and blanked) before the generic pass so the buggy value never becomes
	// an assert.
	reInsteadExample = regexp.MustCompile(assertCall + `(` + assertLit + `)\s+instead\s+of\s+(` + assertLit + `)`)
	reBecomesExample = regexp.MustCompile(`('[^']*'|"[^"]*")\s*(?:becomes|->|→|turns into)\s*('[^']*'|"[^"]*")`)
	reFuncName       = regexp.MustCompile(`(?:function|def)\s+(\w+)\s*\(|\b(\w+)\(\w*\)?\s+(?:in Python|that|should|returning)`)
)

// DeriveAsserts extracts up to 3 executable checks from the prompt.
func DeriveAsserts(prompt string) []string {
	var out []string
	seen := map[string]bool{}
	// "instead of" examples first: assert the corrected value, then blank the
	// span so the generic pass cannot assert the observed buggy value.
	for _, m := range reInsteadExample.FindAllStringSubmatch(prompt, -1) {
		name, args, want := m[1], strings.TrimSpace(m[2]), normalizeLiteral(m[4])
		if name == "" || want == "" {
			continue
		}
		a := fmt.Sprintf("assert %s(%s) == %s", name, args, want)
		if !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	prompt = reInsteadExample.ReplaceAllString(prompt, " ")
	for _, m := range reCallExample.FindAllStringSubmatch(prompt, -1) {
		name, args, want := m[1], strings.TrimSpace(m[2]), normalizeLiteral(m[3])
		if name == "" || want == "" {
			continue
		}
		a := fmt.Sprintf("assert %s(%s) == %s", name, args, want)
		if !seen[a] {
			seen[a] = true
			out = append(out, a)
		}
	}
	// "'x' becomes 'y'" needs a function name from elsewhere in the prompt.
	if len(out) == 0 {
		if fn := functionName(prompt); fn != "" {
			for _, m := range reBecomesExample.FindAllStringSubmatch(prompt, -1) {
				a := fmt.Sprintf("assert %s(%s) == %s", fn, m[1], m[2])
				if !seen[a] {
					seen[a] = true
					out = append(out, a)
				}
			}
		}
	}
	if len(out) > 3 {
		out = out[:3]
	}
	return out
}

func normalizeLiteral(s string) string {
	s = strings.TrimSpace(s)
	// Numbers with thousands separators aren't Python literals.
	if regexp.MustCompile(`^-?\d[\d,]*\.?\d*$`).MatchString(s) {
		return strings.ReplaceAll(strings.ReplaceAll(s, ",", ""), "_", "")
	}
	return s
}

func functionName(prompt string) string {
	if m := reFuncName.FindStringSubmatch(prompt); m != nil {
		if m[1] != "" {
			return m[1]
		}
		return m[2]
	}
	return ""
}
