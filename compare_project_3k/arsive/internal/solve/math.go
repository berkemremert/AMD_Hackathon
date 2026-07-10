package solve

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// EvalExpr evaluates a plain arithmetic expression (+ - * / ^, parentheses,
// unary minus, decimals). The local LLM translates word problems into such
// expressions; evaluating them here costs zero tokens and can't botch the
// arithmetic.
func EvalExpr(expr string) (float64, error) {
	tokens, err := tokenize(expr)
	if err != nil {
		return 0, err
	}
	p := &parser{tokens: tokens}
	v, err := p.parseExpr(0)
	if err != nil {
		return 0, err
	}
	if p.pos != len(p.tokens) {
		return 0, fmt.Errorf("unexpected token %q", p.tokens[p.pos])
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, fmt.Errorf("non-finite result")
	}
	return v, nil
}

// FormatNumber renders a result the way a grader expects: integers without
// decimals, everything else trimmed to at most 6 significant decimals.
func FormatNumber(v float64) string {
	if math.Abs(v-math.Round(v)) < 1e-9 && math.Abs(v) < 1e15 {
		return strconv.FormatFloat(math.Round(v), 'f', -1, 64)
	}
	s := strconv.FormatFloat(v, 'f', 6, 64)
	s = strings.TrimRight(s, "0")
	return strings.TrimRight(s, ".")
}

func tokenize(expr string) ([]string, error) {
	// Normalize common LLM output artifacts.
	r := strings.NewReplacer("×", "*", "÷", "/", ",", "", "$", "", "%", "/100", " ", "")
	expr = r.Replace(expr)
	var tokens []string
	for i := 0; i < len(expr); {
		c := expr[i]
		switch {
		case c >= '0' && c <= '9' || c == '.':
			j := i
			for j < len(expr) && (expr[j] >= '0' && expr[j] <= '9' || expr[j] == '.') {
				j++
			}
			tokens = append(tokens, expr[i:j])
			i = j
		case strings.ContainsRune("+-*/^()", rune(c)):
			tokens = append(tokens, string(c))
			i++
		default:
			return nil, fmt.Errorf("invalid character %q", c)
		}
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty expression")
	}
	return tokens, nil
}

type parser struct {
	tokens []string
	pos    int
}

var precedence = map[string]int{"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}

func (p *parser) parseExpr(minPrec int) (float64, error) {
	left, err := p.parseUnary()
	if err != nil {
		return 0, err
	}
	for p.pos < len(p.tokens) {
		op := p.tokens[p.pos]
		prec, ok := precedence[op]
		if !ok || prec < minPrec {
			break
		}
		p.pos++
		// ^ is right-associative; the rest are left-associative.
		next := prec + 1
		if op == "^" {
			next = prec
		}
		right, err := p.parseExpr(next)
		if err != nil {
			return 0, err
		}
		switch op {
		case "+":
			left += right
		case "-":
			left -= right
		case "*":
			left *= right
		case "/":
			if right == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			left /= right
		case "^":
			left = math.Pow(left, right)
		}
	}
	return left, nil
}

func (p *parser) parseUnary() (float64, error) {
	if p.pos >= len(p.tokens) {
		return 0, fmt.Errorf("unexpected end of expression")
	}
	tok := p.tokens[p.pos]
	switch tok {
	case "-":
		p.pos++
		v, err := p.parseUnary()
		return -v, err
	case "+":
		p.pos++
		return p.parseUnary()
	case "(":
		p.pos++
		v, err := p.parseExpr(0)
		if err != nil {
			return 0, err
		}
		if p.pos >= len(p.tokens) || p.tokens[p.pos] != ")" {
			return 0, fmt.Errorf("missing closing parenthesis")
		}
		p.pos++
		return v, nil
	default:
		v, err := strconv.ParseFloat(tok, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid number %q", tok)
		}
		p.pos++
		return v, nil
	}
}
