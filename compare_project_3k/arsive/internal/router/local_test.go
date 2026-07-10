package router

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"tokenrouter/internal/classify"
	"tokenrouter/internal/llm"
	"tokenrouter/internal/task"
)

// fakeChat serves an OpenAI-compatible chat endpoint returning a fixed body
// and counting calls.
func fakeChat(t *testing.T, content string, calls *atomic.Int64) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": content}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
		})
	}))
}

func testRouter(local, remote *httptest.Server, cats []string) *Router {
	fw := llm.NewFireworks(llm.NewClient(remote.URL, "", 5*time.Second), []string{"gemma-4-26b-a4b-it"})
	opt := Options{RetryBudget: -1, LocalCategories: cats}
	if local != nil {
		opt.Local = llm.NewClient(local.URL, "", 5*time.Second)
	}
	return New(fw, nil, opt)
}

func TestLocalAllowedGating(t *testing.T) {
	var n atomic.Int64
	remote := fakeChat(t, "x", &n)
	defer remote.Close()

	// No local client → never allowed.
	r := testRouter(nil, remote, nil)
	if r.localAllowed(classify.Sentiment) {
		t.Error("nil local client must disable the layer")
	}
	// Local client, no category filter → all allowed.
	local := fakeChat(t, "y", &n)
	defer local.Close()
	r = testRouter(local, remote, nil)
	if !r.localAllowed(classify.Sentiment) || !r.localAllowed(classify.CodeGen) {
		t.Error("empty LocalCategories must allow all categories")
	}
	// Category filter restricts.
	r = testRouter(local, remote, []string{"sentiment", "factual"})
	if !r.localAllowed(classify.Sentiment) || r.localAllowed(classify.CodeGen) {
		t.Error("LocalCategories filter not applied")
	}
}

func TestLocalAnswersWithoutRemoteCall(t *testing.T) {
	var localCalls, remoteCalls atomic.Int64
	local := fakeChat(t, "Positive — clear praise for the product.", &localCalls)
	defer local.Close()
	remote := fakeChat(t, "REMOTE", &remoteCalls)
	defer remote.Close()

	fw := llm.NewFireworks(llm.NewClient(remote.URL, "", 5*time.Second), []string{"gemma-4-26b-a4b-it"})
	r := New(fw, nil, Options{RetryBudget: -1, Local: llm.NewClient(local.URL, "", 5*time.Second)})

	ans := r.Answer(context.Background(), task.Task{ID: "t1",
		Prompt: "Classify the sentiment of this review: 'Absolutely loved it, works perfectly.'"})
	if ans != "Positive — clear praise for the product." {
		t.Fatalf("unexpected answer %q", ans)
	}
	if localCalls.Load() == 0 {
		t.Error("local endpoint was never called")
	}
	if remoteCalls.Load() != 0 {
		t.Errorf("remote must not be called when local passes, got %d calls", remoteCalls.Load())
	}
}

func TestLocalFailureEscalatesToRemote(t *testing.T) {
	var localCalls, remoteCalls atomic.Int64
	// Local returns garbage that fails the sentiment format check.
	local := fakeChat(t, "hmm, hard to say!", &localCalls)
	defer local.Close()
	remote := fakeChat(t, "Negative — sarcastic complaint.", &remoteCalls)
	defer remote.Close()

	fw := llm.NewFireworks(llm.NewClient(remote.URL, "", 5*time.Second), []string{"gemma-4-26b-a4b-it"})
	r := New(fw, nil, Options{RetryBudget: 0, Local: llm.NewClient(local.URL, "", 5*time.Second)})

	ans := r.Answer(context.Background(), task.Task{ID: "t2",
		Prompt: "Classify the sentiment of this review: 'Great, it broke on day one.'"})
	if ans != "Negative — sarcastic complaint." {
		t.Fatalf("unexpected answer %q", ans)
	}
	if localCalls.Load() == 0 || remoteCalls.Load() == 0 {
		t.Errorf("want local attempt then remote escalation, got local=%d remote=%d",
			localCalls.Load(), remoteCalls.Load())
	}
}

func TestThinkingDrainRescue(t *testing.T) {
	// First remote response burns the whole cap inside reasoning_content;
	// the rescue retry with a bigger cap must deliver the real answer.
	var calls atomic.Int64
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"","reasoning_content":"thinking..."},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":40,"total_tokens":45}}`))
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Positive"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":50,"total_tokens":55}}`))
	}))
	defer remote.Close()

	fw := llm.NewFireworks(llm.NewClient(remote.URL, "", 5*time.Second), []string{"gemma-4-26b-a4b-it"})
	r := New(fw, nil, Options{RetryBudget: 0, ReasoningEffort: "none"})

	ans := r.Answer(context.Background(), task.Task{ID: "t4",
		Prompt: "Classify the sentiment of this review: 'Absolutely wonderful device.'"})
	if ans != "Positive" {
		t.Fatalf("drain rescue failed, got %q", ans)
	}
	if calls.Load() != 2 {
		t.Errorf("want exactly 2 calls (drain + rescue), got %d", calls.Load())
	}
}

func TestLocalPALComputesInGo(t *testing.T) {
	var localCalls, remoteCalls atomic.Int64
	local := fakeChat(t, "(240 * 0.85) - 60", &localCalls)
	defer local.Close()
	remote := fakeChat(t, "REMOTE", &remoteCalls)
	defer remote.Close()

	fw := llm.NewFireworks(llm.NewClient(remote.URL, "", 5*time.Second), []string{"gemma-4-26b-a4b-it"})
	r := New(fw, nil, Options{RetryBudget: -1, Local: llm.NewClient(local.URL, "", 5*time.Second)})

	ans := r.Answer(context.Background(), task.Task{ID: "t3",
		Prompt: "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?"})
	if ans != "144" {
		t.Fatalf("want 144 (Go-evaluated), got %q", ans)
	}
	if remoteCalls.Load() != 0 {
		t.Errorf("remote must not be called, got %d", remoteCalls.Load())
	}
}
