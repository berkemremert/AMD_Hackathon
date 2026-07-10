package classify

import "testing"

func TestClassify(t *testing.T) {
	cases := []struct {
		prompt string
		want   Category
	}{
		{"Summarise the following text in one sentence: The quick brown fox...", Summarize},
		{"What is the sentiment of this review: 'Great product, terrible support.'? Explain.", Sentiment},
		{"Extract all people, organizations, and dates from: 'Tim Cook met Apple staff on May 5.'", NER},
		{"Calculate the total cost if a $240 item is discounted by 15% and then taxed 8%.", Math},
		{"What is 15 * (3 + 2)?", Math},
		{"Write a Python function that returns the nth Fibonacci number.", CodeGen},
		{"This function should return the sum but it doesn't work. Fix the bug:\ndef add(a, b):\n    return a - b", CodeDebug},
		{"Three friends sit in a row. Alice is not on the left. Bob sits next to Carol. Who sits where?", Logic},
		{"On an island, knights always tell the truth and knaves always lie. A says: 'We are both knaves.' What are A and B?", Logic},
		{"Five houses in a row are painted red, blue, green, white, and yellow. The red house is immediately left of the blue house. What is the order?", Logic},
		{"Explain how photosynthesis works.", Factual},
		{"What is the capital of Australia?", Factual},
	}
	for _, c := range cases {
		if got := Classify(c.prompt); got != c.want {
			t.Errorf("Classify(%.50q) = %s, want %s", c.prompt, got, c.want)
		}
	}
}
