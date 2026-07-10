// Command smoke validates the live Fireworks path before a real submission,
// so a config problem never burns a leaderboard slot. It uses the SAME client
// code the agent uses (model selection, reasoning_effort, prefix-cache header),
// so what passes here is what the submission will do.
//
// Usage: set FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS (a .env is
// fine — see eval/smoke.sh), then: go run ./cmd/smoke
package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"tokenrouter/internal/config"
	"tokenrouter/internal/llm"
)

func main() {
	cfg := config.FromEnv()
	if cfg.FireworksBaseURL == "" || cfg.FireworksAPIKey == "" || len(cfg.AllowedModels) == 0 {
		fmt.Fprintln(os.Stderr, "FAIL: set FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS first.")
		os.Exit(1)
	}
	fmt.Printf("base_url : %s\n", cfg.FireworksBaseURL)
	fmt.Printf("models   : %v\n", cfg.AllowedModels)
	fmt.Printf("prefix   : %v · reasoning_effort=%q\n\n", cfg.PrefixCache, cfg.ReasoningEffort)

	client := llm.NewClient(cfg.FireworksBaseURL, cfg.FireworksAPIKey, 30*time.Second)
	if cfg.PrefixCache {
		client.Headers = map[string]string{"x-session-affinity": "tokenrouter"}
	}
	fw := llm.NewFireworks(client, cfg.AllowedModels)

	roles := []struct {
		name string
		role llm.Role
	}{{"General", llm.RoleGeneral}, {"Strong", llm.RoleStrong}, {"Code", llm.RoleCode}}

	// 1) Model selection — does Pick() resolve each role to a real allowed model?
	fmt.Println("== model selection (Pick) ==")
	for _, r := range roles {
		m, err := fw.Pick(r.role)
		if err != nil {
			fmt.Printf("  %-8s -> ERROR: %v\n", r.name, err)
			continue
		}
		fmt.Printf("  %-8s -> %s\n", r.name, m)
	}

	ctx := context.Background()
	fail := 0

	// 2) One real call per role, WITH reasoning_effort + prefix-cache header.
	fmt.Println("\n== live calls (reasoning_effort + prefix-cache on) ==")
	for _, r := range roles {
		req := llm.ChatRequest{
			Messages:    []llm.Message{{Role: "user", Content: "Reply with exactly the word: OK"}},
			MaxTokens:   16,
			Temperature: 0,
			Extra:       map[string]any{"reasoning_effort": cfg.ReasoningEffort},
		}
		resp, err := fw.Chat(ctx, r.role, req)
		if err != nil {
			fmt.Printf("  %-8s -> ERROR: %v\n", r.name, err)
			fail++
			continue
		}
		fmt.Printf("  %-8s -> ok  content=%q  usage(p=%d c=%d)\n",
			r.name, truncate(resp.Content), resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}

	// 3) reasoning_effort fallback: if an endpoint rejects the knob, does a
	//    plain call still work? (The router relies on this fallback.)
	fmt.Println("\n== reasoning_effort rejection fallback (plain call) ==")
	{
		resp, err := fw.Chat(ctx, llm.RoleGeneral, llm.ChatRequest{
			Messages:    []llm.Message{{Role: "user", Content: "Reply with exactly the word: OK"}},
			MaxTokens:   16,
			Temperature: 0,
		})
		if err != nil {
			fmt.Printf("  plain call -> ERROR: %v\n", err)
			fail++
		} else {
			fmt.Printf("  plain call -> ok  content=%q\n", truncate(resp.Content))
		}
	}

	fmt.Printf("\n%s\n", fw.Summary())
	if fail > 0 {
		fmt.Fprintf(os.Stderr, "\nFAIL: %d live call(s) errored — fix before submitting.\n", fail)
		os.Exit(1)
	}
	fmt.Println("\nPASS: the live Fireworks path works with our client.")
}

func truncate(s string) string {
	if len(s) > 60 {
		return s[:60] + "..."
	}
	return s
}
