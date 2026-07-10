package router

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"tokenrouter/internal/classify"
	"tokenrouter/internal/llm"
)

// Batchable reports whether a task can share a batched API call: short,
// single-line prompts in a category whose answer is compact enough to parse
// back from a numbered list. Summarization (long input), math/logic (need
// per-task solving/verification), code and NER (multi-line answers) are
// excluded.
func Batchable(cat classify.Category, prompt string) bool {
	switch cat {
	case classify.Sentiment, classify.Factual:
		return !strings.ContainsAny(prompt, "\n") && len(prompt) <= 300
	}
	return false
}

const batchInstruction = " You are given several numbered items. Answer each on its own line as 'N: <answer>' with no blank lines between them."

// AnswerBatch answers a group of same-category prompts in one call, paying the
// system prompt once. Returns (answers, true) only when every item parses
// back; on any shortfall it returns (nil, false) so the caller falls back to
// individual calls (accuracy-safe — a garbled batch never silently drops
// answers).
func (r *Router) AnswerBatch(ctx context.Context, cat classify.Category, prompts []string) ([]string, bool) {
	if len(prompts) == 0 {
		return nil, false
	}
	var sb strings.Builder
	for i, p := range prompts {
		fmt.Fprintf(&sb, "%d. %s\n", i+1, strings.TrimSpace(r.compress(cat, p)))
	}
	role := llm.RoleGeneral
	if cat == classify.Logic {
		role = llm.RoleStrong
	}
	// generic=true suppresses the single-answer grammar (a numbered list
	// must not be constrained to one label); reasoning_effort still applies.
	resp, err := r.chatConstrained(ctx, role, cat, true, llm.ChatRequest{
		Messages:    r.messages(remoteSystem[cat]+batchInstruction, sb.String()),
		MaxTokens:   remoteMaxTokens[cat]*len(prompts) + 32,
		Temperature: 0,
	})
	if err != nil {
		return nil, false
	}
	return parseBatch(resp.Content, len(prompts), cat)
}

var reBatchMarker = regexp.MustCompile(`(?m)^\s*(\d+)[:.)]\s*`)

// parseBatch slices a numbered response into per-item answers. Text between
// marker N and the next marker is item N's answer, tolerating multi-line
// answers. Requires every index 1..n to be present exactly.
func parseBatch(content string, n int, cat classify.Category) ([]string, bool) {
	locs := reBatchMarker.FindAllStringSubmatchIndex(content, -1)
	if len(locs) < n {
		return nil, false
	}
	byNum := map[int]string{}
	for i, loc := range locs {
		num, err := strconv.Atoi(content[loc[2]:loc[3]])
		if err != nil {
			continue
		}
		end := len(content)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		ans := strings.TrimSpace(content[loc[1]:end])
		if ans != "" {
			byNum[num] = ans
		}
	}
	out := make([]string, n)
	for i := 1; i <= n; i++ {
		a, ok := byNum[i]
		if !ok {
			return nil, false
		}
		out[i-1] = postprocess(cat, a)
	}
	return out, true
}
