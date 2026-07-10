// classcheck measures regex-only classification accuracy against the labeled
// eval set, and reports how often the weak-signal LLM fallback would engage.
//
//	go run ./cmd/classcheck eval/tasks.json [eval/expected.json]
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"

	"tokenrouter/internal/classify"
	"tokenrouter/internal/task"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: classcheck <tasks.json> [expected.json]")
		os.Exit(2)
	}
	tasksPath := os.Args[1]
	expectedPath := "eval/expected.json"
	if len(os.Args) > 2 {
		expectedPath = os.Args[2]
	}

	tasks, err := task.Read(tasksPath)
	must(err)
	raw, err := os.ReadFile(expectedPath)
	must(err)
	var expected map[string]struct {
		Category string `json:"category"`
	}
	must(json.Unmarshal(raw, &expected))

	type tally struct{ total, correct, weak int }
	perCat := map[string]*tally{}
	var wrong []string
	weakTotal := 0

	for _, t := range tasks {
		exp, ok := expected[t.ID]
		if !ok {
			continue
		}
		got, score := classify.ClassifyScored(t.Prompt)
		c := perCat[exp.Category]
		if c == nil {
			c = &tally{}
			perCat[exp.Category] = c
		}
		c.total++
		if string(got) == exp.Category {
			c.correct++
		} else {
			wrong = append(wrong, fmt.Sprintf("%s: expected %s, got %s (score %d)", t.ID, exp.Category, got, score))
		}
		if score < 2 {
			c.weak++
			weakTotal++
		}
	}

	cats := make([]string, 0, len(perCat))
	for c := range perCat {
		cats = append(cats, c)
	}
	sort.Strings(cats)
	total, correct := 0, 0
	fmt.Printf("%-12s %8s %8s %6s\n", "category", "correct", "total", "weak")
	for _, c := range cats {
		t := perCat[c]
		total += t.total
		correct += t.correct
		fmt.Printf("%-12s %8d %8d %6d\n", c, t.correct, t.total, t.weak)
	}
	fmt.Printf("\nregex accuracy: %d/%d (%.0f%%), llm-fallback would engage on %d/%d\n",
		correct, total, 100*float64(correct)/float64(total), weakTotal, total)
	if len(wrong) > 0 {
		fmt.Println("\nmisclassified:")
		for _, w := range wrong {
			fmt.Println("  " + w)
		}
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
