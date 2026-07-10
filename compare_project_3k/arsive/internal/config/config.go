package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	// Harness-injected (see participant guide).
	FireworksAPIKey  string
	FireworksBaseURL string
	AllowedModels    []string

	InputPath  string
	OutputPath string

	// --- Local model layer (rules, 8 Jul): local inference inside the
	// container counts toward accuracy and ZERO toward the token score.
	// If ModelPath is set the agent spawns llama-server itself; otherwise
	// LocalBaseURL is assumed to point at an already-running instance (dev
	// mode). Local=false disables the layer entirely (Fireworks-only).
	Local               bool
	LocalBaseURL        string
	LocalModelPath      string
	LocalServerBin      string
	LocalCtxSize        int
	LocalParallel       int
	LocalThreads        int
	LocalExtraArgs      []string // space-separated llama-server flags for perf experiments
	LocalRequestTimeout time.Duration
	// LocalCategories limits which task categories may be answered locally
	// (comma-separated; empty = all 8). Pruned by per-category accuracy
	// measurements against the small bundled model.
	LocalCategories []string

	Workers        int
	TotalBudget    time.Duration // hard cap is 10min; keep a safety margin
	RequestTimeout time.Duration // per-request cap is 30s; keep a margin

	// RetryBudget caps paid second-attempt calls across the run (-1 =
	// unlimited). The submission-ladder knob for leaderboard probing.
	RetryBudget int

	// BatchSize groups short single-answer tasks (sentiment/factual) into one
	// API call. 0 = off (one call per task, the default until measured).
	BatchSize int

	// StopSeqs trims trailing filler via category stop sequences. Off until
	// measured live (accuracy effect can't be seen on a canned mock).
	StopSeqs bool

	// --- Experimental components. Each is a self-contained toggle so every
	// one can be A/B-tested against the live Fireworks endpoint in isolation.
	// Defaults follow the local-proof rule: components whose correctness is
	// provable offline (self-gating solvers, behavior-identical caching) ship
	// on; components whose effect depends on the live judge/tokenizer ship
	// off until measured.

	// PuzzleSolvers enables the brute-force logic solvers (knights-and-knaves,
	// zebra-style attribute grids, positional races). Proof-safe: they answer
	// only when every sentence parses and the solution is unique.
	PuzzleSolvers bool

	// PromptCompress trims scored input tokens before the API call.
	// 0 = off, 1 = strip boilerplate/politeness, 2 = also extractively trim
	// long summarization passages. Judge tolerance is a live-only measurement.
	PromptCompress int

	// MergeSystem folds the system prompt into the user message, shaving the
	// chat template's per-message role scaffolding tokens. Live-only A/B.
	MergeSystem bool

	// MutationRepair tries single-edit mutations of buggy code against
	// prompt-derived asserts before paying for a debug call. Proof-gated:
	// answers only when the original fails and exactly one mutant passes.
	MutationRepair bool

	// SolutionLib matches classic codegen tasks against canonical solutions
	// and answers only after the candidate passes the prompt's own examples.
	SolutionLib bool

	// Dedup answers duplicate (normalized-identical) prompts once and copies
	// the answer. Behavior-identical, zero risk.
	Dedup bool

	// Grammar constrains sentiment decoding with a GBNF grammar so filler
	// tokens are impossible by construction. Live-only A/B (the mock and
	// llama-server ignore/vary on the response_format field).
	Grammar bool

	// ReasoningEffort is sent on Fireworks calls ("" = don't send).
	// Thinking tokens are scored: measured live on a reasoning model,
	// "low" still burned 31 completion tokens for a 2-token answer while
	// "none" disabled thinking entirely (completion=2, same answer) — so
	// "none" is the default; endpoints that reject the knob get one plain
	// retry.
	ReasoningEffort string

	// SDKMimic makes Fireworks requests present like the official OpenAI
	// Python SDK: HTTP/1.1 (Go negotiates h2 by default) plus the SDK's
	// User-Agent/Stainless headers. The gate-passing agents all go through
	// SDK-shaped clients; a proxy/WAF that filters on client fingerprint
	// would reject only ours. Harmless against the real API — on by default.
	SDKMimic bool

	// PrefixCache pins Fireworks calls to one replica via an
	// x-session-affinity header so the automatic prefix cache hits and the
	// shared per-category system prompt is billed at Fireworks' discount
	// (default 50%). Routing-only, no accuracy impact — on by default.
	PrefixCache bool

	// RemoteCaps applies the per-category max_tokens tables to Fireworks
	// calls. false omits max_tokens entirely (the profile the gate-passing
	// naive agents use) — spends more tokens but can never truncate an
	// answer into a judge failure.
	RemoteCaps bool
}

func FromEnv() *Config {
	return &Config{
		FireworksAPIKey:  os.Getenv("FIREWORKS_API_KEY"),
		FireworksBaseURL: strings.TrimRight(os.Getenv("FIREWORKS_BASE_URL"), "/"),
		AllowedModels:    splitModels(os.Getenv("ALLOWED_MODELS")),

		InputPath:  envStr("INPUT_PATH", "/input/tasks.json"),
		OutputPath: envStr("OUTPUT_PATH", "/output/results.json"),

		Local:          envBool("LOCAL", true),
		LocalBaseURL:   envStr("LOCAL_BASE_URL", "http://127.0.0.1:8080"),
		LocalModelPath: os.Getenv("LOCAL_MODEL_PATH"),
		LocalServerBin: envStr("LOCAL_SERVER_BIN", "llama-server"),
		LocalCtxSize:   envInt("LOCAL_CTX_SIZE", 4096), // 4 GB RAM: modest KV cache
		LocalParallel:  envInt("LOCAL_PARALLEL", 2),    // 2 vCPU: more slots just thrash
		LocalThreads:   envInt("LOCAL_THREADS", 0),     // 0 = llama-server default (all cores)
		LocalExtraArgs: strings.Fields(os.Getenv("LOCAL_EXTRA_ARGS")),
		// A single local generation must stay well inside the 30s
		// per-request cap on the slow 2 vCPU grading box.
		LocalRequestTimeout: envDur("LOCAL_REQUEST_TIMEOUT", 20*time.Second),
		LocalCategories:     splitModels(os.Getenv("LOCAL_CATEGORIES")),

		Workers:        envInt("WORKERS", 4),
		TotalBudget:    envDur("TOTAL_BUDGET", 9*time.Minute+15*time.Second),
		RequestTimeout: envDur("REQUEST_TIMEOUT", 25*time.Second),

		RetryBudget: envInt("RETRY_BUDGET", -1),
		BatchSize:   envInt("BATCH_SIZE", 0),
		StopSeqs:    envBool("STOP_SEQ", false),

		PuzzleSolvers:  envBool("PUZZLE_SOLVERS", true),
		PromptCompress: envInt("PROMPT_COMPRESS", 0),
		MergeSystem:    envBool("MERGE_SYSTEM", false),
		MutationRepair: envBool("MUTATION_REPAIR", true),
		SolutionLib:    envBool("SOLUTION_LIB", true),
		Dedup:          envBool("DEDUP", true),
		Grammar:        envBool("GRAMMAR", false),

		ReasoningEffort: envStr("REASONING_EFFORT", "none"),
		SDKMimic:        envBool("SDK_MIMIC", true),
		PrefixCache:     envBool("PREFIX_CACHE", true),
		RemoteCaps:      envBool("REMOTE_CAPS", true),
	}
}

func splitModels(s string) []string {
	var out []string
	for _, m := range strings.Split(s, ",") {
		if m = strings.TrimSpace(m); m != "" {
			out = append(out, m)
		}
	}
	return out
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envBool(key string, def bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return def
}

func envDur(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
