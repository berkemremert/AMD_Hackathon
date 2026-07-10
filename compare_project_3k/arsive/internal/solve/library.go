package solve

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"
)

// Proven solution library: hackathon codegen tasks lean on the classics
// (fibonacci, palindromes, string reversal, primes...). We embed canonical
// implementations — data, not model weights — and answer with one ONLY after
// it passes the asserts derived from the prompt's own examples. The library
// never guesses: no asserts, no proof, no answer. Where conventions differ
// (0- vs 1-indexed fibonacci, strict vs normalized palindromes) we ship one
// variant per convention and let the prompt's examples arbitrate.

type libEntry struct {
	keywords []string // any of these in the lowercased prompt selects the entry
	variants []string // Python templates; %[1]s is the required function name
}

var library = []libEntry{
	{
		keywords: []string{"fibonacci"},
		variants: []string{
			// F(0)=0, F(1)=1 (fib(10) == 55)
			"def %[1]s(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
			// F(1)=1, F(2)=1 (fib(10) == 55 under 1-indexing too, but fib(1)=1)
			"def %[1]s(n):\n    a, b = 1, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return a\n",
		},
	},
	{
		keywords: []string{"factorial"},
		variants: []string{
			"def %[1]s(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n",
		},
	},
	{
		keywords: []string{"palindrome"},
		variants: []string{
			// Normalized: ignore case and non-alphanumerics.
			"def %[1]s(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n",
			// Strict comparison.
			"def %[1]s(s):\n    return s == s[::-1]\n",
		},
	},
	{
		keywords: []string{"reverse the words", "reverse word", "word order", "words in reverse", "order of words", "order of the words"},
		variants: []string{
			"def %[1]s(s):\n    return ' '.join(s.split()[::-1])\n",
		},
	},
	{
		keywords: []string{"reverse a string", "reverse the string", "reversed string", "string reversal"},
		variants: []string{
			"def %[1]s(s):\n    return s[::-1]\n",
		},
	},
	{
		keywords: []string{"prime"},
		variants: []string{
			"def %[1]s(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n %% i == 0:\n            return False\n        i += 1\n    return True\n",
		},
	},
	{
		keywords: []string{"greatest common divisor", "gcd"},
		variants: []string{
			"def %[1]s(a, b):\n    while b:\n        a, b = b, a %% b\n    return a\n",
		},
	},
	{
		keywords: []string{"vowel"},
		variants: []string{
			"def %[1]s(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
		},
	},
	{
		keywords: []string{"anagram"},
		variants: []string{
			"def %[1]s(a, b):\n    return sorted(a.lower()) == sorted(b.lower())\n",
			"def %[1]s(a, b):\n    return sorted(a) == sorted(b)\n",
		},
	},
	{
		keywords: []string{"sum of the digits", "digit sum", "sum of digits"},
		variants: []string{
			"def %[1]s(n):\n    return sum(int(d) for d in str(abs(n)))\n",
		},
	},
	{
		keywords: []string{"bracket", "parenthes", "balanced"},
		variants: []string{
			"def %[1]s(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for c in s:\n        if c in '([{':\n            stack.append(c)\n        elif c in pairs:\n            if not stack or stack.pop() != pairs[c]:\n                return False\n    return not stack\n",
		},
	},
	{
		keywords: []string{"fizzbuzz", "fizz buzz"},
		variants: []string{
			"def %[1]s(n):\n    out = []\n    for i in range(1, n + 1):\n        if i %% 15 == 0:\n            out.append('FizzBuzz')\n        elif i %% 3 == 0:\n            out.append('Fizz')\n        elif i %% 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n",
			// Single-value convention: fizzbuzz(3) == 'Fizz'.
			"def %[1]s(i):\n    if i %% 15 == 0:\n        return 'FizzBuzz'\n    if i %% 3 == 0:\n        return 'Fizz'\n    if i %% 5 == 0:\n        return 'Buzz'\n    return str(i)\n",
		},
	},
}

var (
	reLibAssertName = regexp.MustCompile(`^assert (\w+)\(`)
	reLibOtherLang  = regexp.MustCompile(`(?i)\b(?:javascript|typescript|c\+\+|rust|ruby|java)\b|\bin go\b|\bgolang\b`)
)

const libRunTimeout = 3 * time.Second

// LibrarySolve answers a classic codegen task from the embedded library,
// proven against the prompt's own examples. ok=false defers to the model.
func LibrarySolve(ctx context.Context, prompt string) (string, bool) {
	lp := strings.ToLower(prompt)
	// Only Python answers live in the library; another named language → defer.
	if reLibOtherLang.MatchString(prompt) && !strings.Contains(lp, "python") {
		return "", false
	}

	// Proof material: the prompt's own examples, and one agreed function name.
	asserts := DeriveAsserts(prompt)
	if len(asserts) == 0 {
		return "", false
	}
	fn := ""
	for _, a := range asserts {
		m := reLibAssertName.FindStringSubmatch(a)
		if m == nil || (fn != "" && fn != m[1]) {
			return "", false
		}
		fn = m[1]
	}
	if fn == "" {
		return "", false
	}

	for _, entry := range library {
		if !libMatches(lp, entry.keywords) {
			continue
		}
		for _, tpl := range entry.variants {
			code := fmt.Sprintf(tpl, fn)
			res, err := RunPython(ctx, code+"\n"+strings.Join(asserts, "\n"), libRunTimeout)
			if err == nil && !res.TimedOut && res.ExitCode == 0 {
				return strings.TrimSpace(code), true // proven on the prompt's examples
			}
		}
	}
	return "", false
}

func libMatches(lp string, keywords []string) bool {
	for _, k := range keywords {
		if strings.Contains(lp, k) {
			return true
		}
	}
	return false
}
