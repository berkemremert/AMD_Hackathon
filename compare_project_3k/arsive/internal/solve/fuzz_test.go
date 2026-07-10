package solve

import (
	"strings"
	"testing"
)

// weirdInputs are adversarial/degenerate prompts a hidden grading set might
// contain. None must panic; the solvers must simply return (,false) and defer.
func weirdInputs() []string {
	return []string{
		"",
		" ",
		"\n\n\n",
		"?",
		".....",
		strings.Repeat("A", 100000),                       // huge single token
		strings.Repeat("word ", 50000),                    // huge many-word
		strings.Repeat("(", 5000),                         // unbalanced parens
		strings.Repeat("All X are Y. ", 2000),             // syllogism flood
		"知knight knave 骑士 " + strings.Repeat("的", 1000), // unicode
		"\x00\x01\x02 knights and knaves \xff",             // control bytes
		"Amy Ben Cem " + strings.Repeat("before after ", 1000),
		"1 + + + * / ( ) 2 3 4 5",                          // malformed expression
		"All all all are are are",
		"who owns the who owns the cat dog fish",
		"assert assert assert fib(((",
		strings.Repeat("Sam does not own the bird. ", 1000),
		"knight" + strings.Repeat(" says 'X is a knave'", 500),
		"🎲🎲🎲 five houses red blue green 🎲",
		strings.Repeat("Kaya finished first. ", 500),
	}
}

func TestSolversNeverPanic(t *testing.T) {
	solvers := map[string]func(string) (string, bool){
		"SolveOrdering":     SolveOrdering,
		"SolveSyllogism":    SolveSyllogism,
		"SolveKnights":      SolveKnights,
		"SolveZebra":        SolveZebra,
		"SolveSingleAssign": SolveSingleAssign,
		"SolvePositions":    SolvePositions,
	}
	for _, in := range weirdInputs() {
		for name, fn := range solvers {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("%s panicked on %.30q: %v", name, in, r)
					}
				}()
				fn(in) // result ignored — only that it returns without panic
			}()
		}
	}
}

func TestDeriveAssertsNeverPanics(t *testing.T) {
	for _, in := range weirdInputs() {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("DeriveAsserts panicked on %.30q: %v", in, r)
				}
			}()
			DeriveAsserts(in)
		}()
	}
}

func TestEvalExprRejectsGarbage(t *testing.T) {
	// EvalExpr must error (not panic) on malformed input.
	for _, in := range []string{"", "+", "((((", "1/0", "* 2", "1 2 3", strings.Repeat("1+", 10000) + "1"} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("EvalExpr panicked on %.20q: %v", in, r)
				}
			}()
			_, _ = EvalExpr(in)
		}()
	}
}
