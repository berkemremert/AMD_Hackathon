package solve

import "testing"

func TestDeriveAsserts(t *testing.T) {
	cases := []struct {
		prompt string
		want   []string
	}{
		{
			"Write is_palindrome(s). Examples: is_palindrome('Race car') is True, is_palindrome('hello') is False.",
			[]string{"assert is_palindrome('Race car') == True", "assert is_palindrome('hello') == False"},
		},
		{
			"parse_duration('2h30m') == 9000 and parse_duration('45s') returns 45",
			[]string{"assert parse_duration('2h30m') == 9000", "assert parse_duration('45s') == 45"},
		},
		{
			"Write a Python function reverse_words(sentence) that reverses word order. Example: 'hello world foo' becomes 'foo world hello'.",
			[]string{"assert reverse_words('hello world foo') == 'foo world hello'"},
		},
		{
			"Explain how photosynthesis works.",
			nil,
		},
		{
			// Debug phrasing: assert the corrected value, never the buggy one.
			"This is buggy: factorial(5) returns 24 instead of 120. Fix it.",
			[]string{"assert factorial(5) == 120"},
		},
		{
			"Write square(n). square(4) should return 16 and square(9) must be 81.",
			[]string{"assert square(4) == 16", "assert square(9) == 81"},
		},
	}
	for _, c := range cases {
		got := DeriveAsserts(c.prompt)
		if len(got) != len(c.want) {
			t.Errorf("DeriveAsserts(%.40q) = %v, want %v", c.prompt, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("DeriveAsserts(%.40q)[%d] = %q, want %q", c.prompt, i, got[i], c.want[i])
			}
		}
	}
}
