package solve

import (
	"context"
	"os/exec"
	"strings"
	"testing"
)

func needPython(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not available")
	}
}

// --- mutation repair ---

func TestRepairPythonComparisonBug(t *testing.T) {
	needPython(t)
	// Classic off-by-one comparison: factorial(5) should be 120 but "< n"
	// stops one multiply early... actually range(1, n) omits n itself.
	buggy := "def fact(n):\n    out = 1\n    for i in range(1, n):\n        out *= i\n    return out\n"
	asserts := []string{"assert fact(5) == 120", "assert fact(3) == 6"}
	fixed, desc, ok := RepairPython(context.Background(), buggy, asserts)
	if !ok {
		t.Fatal("expected a repair")
	}
	if !strings.Contains(fixed, "range(1, n + 1)") && !strings.Contains(fixed, "range(1, n +1)") {
		t.Errorf("unexpected fix: %q (%s)", fixed, desc)
	}
}

func TestRepairPythonOperatorBug(t *testing.T) {
	needPython(t)
	buggy := "def bigger(a, b):\n    if a - b:\n        return a\n    return b\n"
	// A single flip can't be proven unambiguously here (several mutants may
	// pass) — the point of this case is a definite single-token bug:
	buggy = "def total(a, b):\n    return a - b\n"
	asserts := []string{"assert total(2, 3) == 5", "assert total(10, 1) == 11"}
	fixed, _, ok := RepairPython(context.Background(), buggy, asserts)
	if !ok {
		t.Fatal("expected a repair")
	}
	if !strings.Contains(fixed, "a + b") {
		t.Errorf("unexpected fix: %q", fixed)
	}
}

func TestRepairPythonDefersWhenOriginalPasses(t *testing.T) {
	needPython(t)
	good := "def total(a, b):\n    return a + b\n"
	if _, _, ok := RepairPython(context.Background(), good, []string{"assert total(2, 3) == 5"}); ok {
		t.Fatal("must defer when the 'bug' is not provable")
	}
}

func TestRepairPythonDefersWithoutAsserts(t *testing.T) {
	if _, _, ok := RepairPython(context.Background(), "def f():\n    return 1\n", nil); ok {
		t.Fatal("must defer without proof material")
	}
}

func TestExtractPromptCode(t *testing.T) {
	prompt := "This function is wrong. Fix it:\n\ndef f(x):\n    return x - 1\n"
	code := ExtractPromptCode(prompt)
	if !strings.HasPrefix(code, "def f(x):") {
		t.Errorf("got %q", code)
	}
	if ExtractPromptCode("no code here at all") != "" {
		t.Error("prose must yield no code")
	}
}

// --- solution library ---

func TestLibraryFibonacci(t *testing.T) {
	needPython(t)
	prompt := "Write a Python function fib(n) that returns the nth Fibonacci number. Example: fib(10) returns 55."
	code, ok := LibrarySolve(context.Background(), prompt)
	if !ok {
		t.Fatal("expected a library hit")
	}
	if !strings.Contains(code, "def fib(") {
		t.Errorf("wrong function name: %q", code)
	}
}

func TestLibraryPalindromeNormalized(t *testing.T) {
	needPython(t)
	// The example forces the case/punctuation-insensitive variant.
	prompt := "Write a Python function is_pal(s) to check for a palindrome. " +
		"Example: is_pal('A man, a plan, a canal: Panama!') returns True."
	code, ok := LibrarySolve(context.Background(), prompt)
	if !ok {
		t.Fatal("expected a library hit")
	}
	if !strings.Contains(code, "isalnum") {
		t.Errorf("examples should have selected the normalized variant: %q", code)
	}
}

func TestLibraryDefers(t *testing.T) {
	needPython(t)
	cases := []string{
		// No examples → no proof → defer.
		"Write a Python function fib(n) that returns the nth Fibonacci number.",
		// Another language → defer.
		"Write a JavaScript function fib(n) for Fibonacci. Example: fib(10) returns 55.",
		// Examples contradict every variant → defer.
		"Write a Python function fib(n) for Fibonacci. Example: fib(10) returns 89.",
	}
	for _, p := range cases {
		if code, ok := LibrarySolve(context.Background(), p); ok {
			t.Errorf("expected defer for %q, got %q", p, code)
		}
	}
}
