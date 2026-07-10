package solve

import "testing"

func TestEvalExpr(t *testing.T) {
	cases := []struct {
		expr string
		want string
	}{
		{"2+2", "4"},
		{"15*(3+2)", "75"},
		{"0.15*240+18", "54"},
		{"100/4/5", "5"},
		{"2^10", "1024"},
		{"-(3+4)*2", "-14"},
		{"1,200*3", "3600"},
		{"50%", "0.5"},
		{"$1,500.50 + $200", "1700.5"},
		{"10/3", "3.333333"},
		{"2 ^ 3 ^ 2", "512"}, // right-associative
	}
	for _, c := range cases {
		v, err := EvalExpr(c.expr)
		if err != nil {
			t.Errorf("EvalExpr(%q): %v", c.expr, err)
			continue
		}
		if got := FormatNumber(v); got != c.want {
			t.Errorf("EvalExpr(%q) = %s, want %s", c.expr, got, c.want)
		}
	}
}

func TestEvalExprErrors(t *testing.T) {
	for _, expr := range []string{"", "abc", "2+", "(3", "1/0", "2**3"} {
		if _, err := EvalExpr(expr); err == nil {
			t.Errorf("EvalExpr(%q): expected error", expr)
		}
	}
}
