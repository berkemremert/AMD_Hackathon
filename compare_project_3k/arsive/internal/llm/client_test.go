package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestChatRetriesTransientErrors(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits < 3 {
			w.WriteHeader(http.StatusTooManyRequests) // rate-limited twice
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", 10*time.Second)
	resp, err := c.Chat(context.Background(), ChatRequest{Model: "m",
		Messages: []Message{{Role: "user", Content: "hi"}}})
	if err != nil || resp.Content != "ok" {
		t.Fatalf("want recovery after 429s, got resp=%v err=%v", resp, err)
	}
	if hits != 3 {
		t.Errorf("want 3 attempts, got %d", hits)
	}

	// A 4xx that isn't a rate limit must NOT be retried.
	hits = 0
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv2.Close()
	c2 := NewClient(srv2.URL, "", 10*time.Second)
	if _, err := c2.Chat(context.Background(), ChatRequest{Model: "m",
		Messages: []Message{{Role: "user", Content: "hi"}}}); err == nil {
		t.Fatal("want error on 404")
	}
	if hits != 1 {
		t.Errorf("404 must not be retried, got %d attempts", hits)
	}
}

func TestChatBodySerialization(t *testing.T) {
	var got map[string]any
	var gotAffinity string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAffinity = r.Header.Get("x-session-affinity")
		raw, _ := io.ReadAll(r.Body)
		var parsed map[string]any
		_ = json.Unmarshal(raw, &parsed)
		got = parsed // fresh map per request — stale keys must not linger
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", 5*time.Second)
	c.Headers = map[string]string{"x-session-affinity": "tokenrouter"}
	_, err := c.Chat(context.Background(), ChatRequest{
		Model:          "m",
		Messages:       []Message{{Role: "user", Content: "hi"}},
		MaxTokens:      7,
		Stop:           []string{"\n\n"},
		ResponseFormat: map[string]any{"type": "grammar", "grammar": "root ::= \"x\""},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got["max_tokens"] != float64(7) {
		t.Errorf("max_tokens = %v", got["max_tokens"])
	}
	if _, ok := got["stop"]; !ok {
		t.Error("stop missing from body")
	}
	rf, ok := got["response_format"].(map[string]any)
	if !ok || rf["type"] != "grammar" {
		t.Errorf("response_format missing or wrong: %v", got["response_format"])
	}
	if gotAffinity != "tokenrouter" {
		t.Errorf("x-session-affinity header = %q, want tokenrouter", gotAffinity)
	}

	// And absent when unset — a stray null field could 400 on strict servers.
	_, err = c.Chat(context.Background(), ChatRequest{
		Model:    "m",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, present := got["response_format"]; present {
		t.Error("response_format must be omitted when unset")
	}
	if _, present := got["stop"]; present {
		t.Error("stop must be omitted when unset")
	}
}
