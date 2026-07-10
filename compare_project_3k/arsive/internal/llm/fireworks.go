package llm

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
)

// Role picks which class of remote model a task escalates to.
type Role int

const (
	RoleGeneral Role = iota // cheap/fast default: gemma-4-26b-a4b-it
	RoleStrong              // harder factual/logic/math: gemma-4-31b-it
	RoleCode                // last-resort code accuracy: kimi-k2p7-code
)

// Fireworks wraps the harness-provided endpoint with model selection from
// ALLOWED_MODELS and global token accounting. Every recorded token counts
// toward the leaderboard score, so calls go through Chat here and nowhere else.
type Fireworks struct {
	client *Client
	models []string

	promptTokens     atomic.Int64
	completionTokens atomic.Int64
	calls            atomic.Int64
}

func NewFireworks(client *Client, allowedModels []string) *Fireworks {
	return &Fireworks{client: client, models: allowedModels}
}

// Pick maps a role to an allowed model ID by substring, falling back through
// preference lists so we never emit a model outside ALLOWED_MODELS.
func (f *Fireworks) Pick(role Role) (string, error) {
	var prefs []string
	switch role {
	case RoleGeneral:
		prefs = []string{"a4b", "gemma", "minimax"}
	case RoleStrong:
		prefs = []string{"gemma-4-31b-it", "31b", "gemma", "minimax"}
	case RoleCode:
		prefs = []string{"code", "gemma-4-31b-it", "gemma", "minimax"}
	}
	for _, p := range prefs {
		for _, m := range f.models {
			ml := strings.ToLower(m)
			// Prefer the non-quantized 31b when both variants are allowed.
			if role == RoleStrong && p == "gemma-4-31b-it" && strings.Contains(ml, "nvfp4") {
				continue
			}
			if strings.Contains(ml, p) {
				return m, nil
			}
		}
	}
	if len(f.models) > 0 {
		return f.models[0], nil
	}
	return "", fmt.Errorf("no allowed models configured")
}

func (f *Fireworks) Chat(ctx context.Context, role Role, req ChatRequest) (*ChatResponse, error) {
	model, err := f.Pick(role)
	if err != nil {
		return nil, err
	}
	req.Model = model
	resp, err := f.client.Chat(ctx, req)
	if err != nil {
		return nil, err
	}
	f.calls.Add(1)
	f.promptTokens.Add(int64(resp.Usage.PromptTokens))
	f.completionTokens.Add(int64(resp.Usage.CompletionTokens))
	return resp, nil
}

func (f *Fireworks) Summary() string {
	p, c := f.promptTokens.Load(), f.completionTokens.Load()
	return fmt.Sprintf("fireworks: %d calls, %d prompt + %d completion = %d tokens",
		f.calls.Load(), p, c, p+c)
}
