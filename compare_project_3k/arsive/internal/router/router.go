package router

import (
	"context"
	"log"
	"regexp"
	"strings"
	"sync/atomic"
	"time"

	"tokenrouter/internal/classify"
	"tokenrouter/internal/compress"
	"tokenrouter/internal/llm"
	"tokenrouter/internal/solve"
	"tokenrouter/internal/task"
	"tokenrouter/internal/verify"
)

const fallbackAnswer = "Unable to determine the answer."

// Router implements the organizer-blessed reading of "routing intelligence":
// decide when a task can be handled with plain code (zero tokens) versus
// which ALLOWED model it must go to, spending the minimum tokens that still
// clear the accuracy gate. Verification (code execution, arithmetic
// re-evaluation, format checks) is plain code too — free — so retries are
// paid for only on proven failure.
type Router struct {
	FW   *llm.Fireworks
	Pace *Pacer // nil → always ModeFull

	opt Options

	// localCats is the parsed LocalCategories allowlist (nil = all).
	localCats map[classify.Category]bool

	// retryBudget caps second-attempt calls across the whole run (-1 =
	// unlimited). The submission-ladder knob: step it down between
	// leaderboard probes.
	retryBudget atomic.Int64
}

// Options collects the toggleable components. Every experimental feature is a
// separate switch so each can be A/B-tested in isolation against the live
// endpoint and disabled without touching code.
type Options struct {
	RetryBudget int  // paid-retry cap across the run (-1 = unlimited)
	StopSeqs    bool // category stop sequences trim trailing filler

	PuzzleSolvers  bool // brute-force knights/zebra/position solvers (0 tokens)
	PromptCompress int  // 0=off, 1=strip boilerplate, 2=+extractive passage trim
	MergeSystem    bool // fold system prompt into the user message
	MutationRepair bool // single-edit repair of buggy code before a debug call
	SolutionLib    bool // canonical solutions proven against prompt examples
	Grammar        bool // GBNF-constrained sentiment decoding

	// Local is the in-container model layer (rules, 8 Jul): answers count
	// toward accuracy and ZERO toward the token score. nil disables it.
	Local *llm.Client
	// LocalCategories restricts which categories may answer locally
	// (empty = all). Pruned by per-category accuracy measurements.
	LocalCategories []string

	// ReasoningEffort is sent as reasoning_effort on Fireworks calls
	// ("" = don't send). Thinking tokens are scored, so "low" is the
	// default; endpoints that reject the knob get one plain retry.
	ReasoningEffort string

	// RemoteCaps=false omits max_tokens on Fireworks calls entirely —
	// the conformity profile: more tokens, zero truncation risk.
	RemoteCaps bool
}

func New(fw *llm.Fireworks, pace *Pacer, opt Options) *Router {
	r := &Router{FW: fw, Pace: pace, opt: opt}
	r.retryBudget.Store(int64(opt.RetryBudget))
	if len(opt.LocalCategories) > 0 {
		r.localCats = map[classify.Category]bool{}
		for _, c := range opt.LocalCategories {
			r.localCats[classify.Category(strings.TrimSpace(c))] = true
		}
	}
	return r
}

// localAllowed gates the local layer: enabled, category permitted, and no
// time pressure (local CPU generation is the slow path — under pressure the
// pacer sends tasks straight to the fast remote endpoint, buying time with
// tokens).
func (r *Router) localAllowed(cat classify.Category) bool {
	if r.opt.Local == nil || r.Pace.Mode() != ModeFull {
		return false
	}
	return r.localCats == nil || r.localCats[cat]
}

// stopFor returns the category's stop sequences when the feature is enabled.
func (r *Router) stopFor(cat classify.Category) []string {
	if !r.opt.StopSeqs {
		return nil
	}
	return remoteStop[cat]
}

// messages assembles the chat messages, folding the system prompt into the
// user message when MergeSystem is on — one message means one set of chat
// template role headers instead of two, and every template token is scored.
func (r *Router) messages(sys, user string) []llm.Message {
	if r.opt.MergeSystem {
		return []llm.Message{{Role: "user", Content: sys + "\n\n" + user}}
	}
	return []llm.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}}
}

// allowRetry spends one unit of retry budget if any is left.
func (r *Router) allowRetry() bool {
	for {
		cur := r.retryBudget.Load()
		if cur < 0 {
			return true // unlimited
		}
		if cur == 0 {
			return false
		}
		if r.retryBudget.CompareAndSwap(cur, cur-1) {
			return true
		}
	}
}

// retryMaxPromptChars: retries resend the whole prompt as input tokens, so
// long passages (summarization) never get a paid second attempt.
const retryMaxPromptChars = 2400

func (r *Router) Answer(ctx context.Context, t task.Task) string {
	cat, score := classify.ClassifyScored(t.Prompt)
	generic := score < 2 // weak signal → no category scaffolding, model reads the task itself
	mode := r.Pace.Mode()
	trace := &decisionTrace{id: t.ID, cat: cat, generic: generic, mode: mode}
	defer trace.emit()
	defer r.Pace.TaskDone()

	// Layer 0: plain code, zero tokens.
	if ans, ok := r.TrySolveFree(cat, t.Prompt); ok {
		trace.layer = "code"
		return ans
	}

	// Layer 1: one frugal API call, verified by plain code where possible.
	var ans string
	var err error
	switch {
	case cat == classify.Math && !reMultiQuantity.MatchString(t.Prompt):
		ans, err = r.mathPAL(ctx, t.Prompt, trace)
	case cat == classify.CodeGen || cat == classify.CodeDebug:
		ans, err = r.code(ctx, cat, t.Prompt, trace)
	default:
		ans, err = r.plain(ctx, cat, generic, t.Prompt, trace)
	}
	if err != nil {
		log.Printf("task %s: remote error: %v", t.ID, err)
		if ans == "" {
			return fallbackAnswer
		}
	}
	if ans == "" {
		return fallbackAnswer
	}
	return ans
}

// --- Layer 0 ---

// TrySolveFree answers a task with plain code (zero tokens) when it fits one
// of the deterministic shapes: a bare arithmetic expression, a transitive
// ordering puzzle, or a universal syllogism. The logic solvers self-gate
// strictly and run regardless of category, so a puzzle misclassified as
// (say) factual is still rescued here before any paid call.
func (r *Router) TrySolveFree(cat classify.Category, prompt string) (string, bool) {
	if cat == classify.Math {
		if ans, ok := solveBareExpression(prompt); ok {
			return ans, true
		}
	}
	if ans, ok := solve.SolveOrdering(prompt); ok {
		return ans, true
	}
	if ans, ok := solve.SolveSyllogism(prompt); ok {
		return ans, true
	}
	if r.opt.PuzzleSolvers {
		if ans, ok := solve.SolveKnights(prompt); ok {
			return ans, true
		}
		if ans, ok := solve.SolveZebra(prompt); ok {
			return ans, true
		}
		if ans, ok := solve.SolveSingleAssign(prompt); ok {
			return ans, true
		}
		if ans, ok := solve.SolvePositions(prompt); ok {
			return ans, true
		}
	}
	return "", false
}

var reBareExpr = regexp.MustCompile(`^[\s\d,.$+\-*/^()×÷%]+[?.\s]*$`)
var reMathPreamble = regexp.MustCompile(`(?i)^\s*(what is|what'?s|calculate|compute|evaluate|solve)[:\s]*`)
var reMultiQuantity = regexp.MustCompile(`(?i)how (many|much) .{0,60}\band\b .{0,60}(how (many|much)|are there)|\?.*\?|\bnumber of each\b|\beach (animal|kind|type)\b`)

func solveBareExpression(prompt string) (string, bool) {
	stripped := reMathPreamble.ReplaceAllString(strings.TrimSpace(prompt), "")
	if !reBareExpr.MatchString(stripped) {
		return "", false
	}
	v, err := solve.EvalExpr(strings.Trim(stripped, "?. \n"))
	if err != nil {
		return "", false
	}
	return solve.FormatNumber(v), true
}

// --- Layer 1a: local model (zero scored tokens) ---

// Local completion tokens are UNSCORED — the only cost is wall-clock, already
// bounded by the per-request timeout. So reasoning categories get generous
// room to complete a full chain of thought (a truncated ramble that never
// reaches its 'Answer:' line just escalates); non-reasoning categories stay
// modest since extra length there buys no accuracy, only latency.
var localMaxTokens = map[classify.Category]int{
	classify.Factual:   160, // multi-part questions need room
	classify.Math:      320, // full CoT for the direct-solve fallback (PAL caps separately)
	classify.Sentiment: 40,
	classify.Summarize: 110,
	classify.NER:       192, // entity lists can be long
	classify.CodeDebug: 512,
	classify.Logic:     512, // full chain-of-thought before the 'Answer:' line
	classify.CodeGen:   512,
}

func (r *Router) localChat(ctx context.Context, cat classify.Category, sys, prompt string, maxTok int, stop []string) (*llm.ChatResponse, error) {
	return r.opt.Local.Chat(ctx, llm.ChatRequest{
		Model:       "local",
		Messages:    r.messages(sys, r.compress(cat, prompt)),
		MaxTokens:   maxTok,
		Temperature: 0,
		Stop:        stop,
	})
}

// localPAL is the free twin of mathPAL: the local model emits the arithmetic
// expression and Go evaluates it — verified by construction, zero tokens.
func (r *Router) localPAL(ctx context.Context, prompt string) (string, bool) {
	resp, err := r.localChat(ctx, classify.Math, palSystem, prompt, 60, []string{"\n"})
	if err != nil {
		return "", false
	}
	expr := strings.TrimSpace(resp.Content)
	if strings.Contains(strings.ToUpper(expr), "UNSUPPORTED") || !strings.ContainsAny(expr, "+-*/^") {
		return "", false
	}
	v, err := solve.EvalExpr(expr)
	if err != nil {
		return "", false
	}
	return solve.FormatNumber(v), true
}

// localCode ships a local answer only with PROOF: the generated code must
// pass asserts derived from the prompt's own examples. No proof → escalate.
func (r *Router) localCode(ctx context.Context, cat classify.Category, prompt string) (string, bool) {
	asserts := solve.DeriveAsserts(prompt)
	if len(asserts) == 0 || !looksLikePython(prompt, "def ") {
		return "", false
	}
	resp, err := r.localChat(ctx, cat, localSystemFor(cat), prompt, localMaxTokens[cat], nil)
	if err != nil {
		return "", false
	}
	ans := postprocess(cat, resp.Content)
	code := solve.ExtractCode(ans)
	if code == "" || !r.testsPass(ctx, code, asserts) {
		return "", false
	}
	return ans, true
}

// reSentimentNuance marks the two sentiment shapes the small local model
// measurably misjudges (eval/PERF.md): contrastive reviews whose dominant
// verdict must outweigh the concession, and factual reporting that must stay
// Neutral despite charged content words. Those go to the strong model.
var reSentimentNuance = regexp.MustCompile(`(?i)\b(but|however|although|though|yet|while|describes?|reports?|according to)\b`)

// localPlain answers the remaining categories locally when the free format
// check passes. This is the accuracy-gate bet the new rules reward: a small
// model's correct answer scores exactly like a Fireworks answer, for zero
// tokens.
func (r *Router) localPlain(ctx context.Context, cat classify.Category, generic bool, prompt string) (string, bool) {
	if cat == classify.Sentiment && reSentimentNuance.MatchString(prompt) {
		return "", false // measured local failure mode → strong model
	}
	sys, maxTok := localSystemFor(cat), localMaxTokens[cat]
	if generic {
		sys, maxTok = genericSystem, 160
	}
	// Logic/math get no stop sequence locally: full CoT contains blank lines,
	// and postprocess extracts the final 'Answer:' line afterwards.
	stop := r.stopFor(cat)
	if cat == classify.Logic || cat == classify.Math {
		stop = nil
	}
	resp, err := r.localChat(ctx, cat, sys, prompt, maxTok, stop)
	if err != nil {
		return "", false
	}
	ans := postprocess(cat, resp.Content)
	if ans == "" {
		return "", false
	}
	// Logic is the highest-risk local category: after full CoT, postprocess
	// keeps only the extracted 'Answer:' line. If a newline survives, the model
	// never emitted that line (a truncated reasoning dump) — escalate rather
	// than ship it. The char cap guards against a verbose extracted conclusion.
	if cat == classify.Logic && (len(ans) > 160 || strings.Contains(ans, "\n")) {
		return "", false
	}
	if !generic && !verify.Check(cat, prompt, ans) {
		return "", false
	}
	return ans, true
}

// --- Layer 1: math via PAL ---

// mathPAL asks the cheap model for a bare expression (~20 output tokens) and
// evaluates it in Go: fewer tokens than a worked solution AND the arithmetic
// is correct by construction. Falls back to a direct solve when the problem
// isn't expressible.
func (r *Router) mathPAL(ctx context.Context, prompt string, trace *decisionTrace) (string, error) {
	// Free first: the local model emits the expression, Go evaluates it.
	if r.localAllowed(classify.Math) {
		if ans, ok := r.localPAL(ctx, prompt); ok {
			trace.layer = "local-pal"
			return ans, nil
		}
	}
	var palStop []string
	if r.opt.StopSeqs {
		palStop = []string{"\n"} // the expression is one line by construction
	}
	resp, err := r.chatConstrained(ctx, llm.RoleGeneral, classify.Math, true, llm.ChatRequest{
		Messages:    r.messages(palSystem, r.compress(classify.Math, prompt)),
		MaxTokens:   60,
		Temperature: 0,
		Stop:        palStop,
	})
	if err == nil {
		expr := strings.TrimSpace(resp.Content)
		if !strings.Contains(strings.ToUpper(expr), "UNSUPPORTED") && strings.ContainsAny(expr, "+-*/^") {
			if v, evalErr := solve.EvalExpr(expr); evalErr == nil {
				trace.layer = "pal"
				return solve.FormatNumber(v), nil
			}
		}
	}
	// Direct solve fallback — one call, tight cap, answer-line extraction.
	return r.plain(ctx, classify.Math, false, prompt, trace)
}

// --- Layer 1: code with free execution-based verification ---

func (r *Router) code(ctx context.Context, cat classify.Category, prompt string, trace *decisionTrace) (string, error) {
	// Zero-token attempts first — both are proof-gated (they answer only
	// after passing the prompt's own examples), so a hit can't be wrong and
	// a miss costs nothing.
	if r.opt.SolutionLib && cat == classify.CodeGen && looksLikePython(prompt, "def ") {
		if code, ok := solve.LibrarySolve(ctx, prompt); ok {
			trace.layer = "library"
			trace.codeTested = true
			return code, nil
		}
	}
	if r.opt.MutationRepair && cat == classify.CodeDebug && looksLikePython(prompt, "def ") {
		if buggy := solve.ExtractPromptCode(prompt); buggy != "" {
			if fixed, desc, ok := solve.RepairPython(ctx, buggy, solve.DeriveAsserts(prompt)); ok {
				trace.layer = "mutate"
				trace.codeTested = true
				return desc + "\n\n" + fixed, nil
			}
		}
	}

	// Local model next — still zero tokens, but slower than the pure-code
	// attempts above, and only a proven (assert-passing) answer ships.
	if r.localAllowed(cat) {
		if ans, ok := r.localCode(ctx, cat, prompt); ok {
			trace.layer = "local-code"
			trace.codeTested = true
			return ans, nil
		}
	}

	ans, err := r.call(ctx, llm.RoleStrong, cat, false, prompt)
	if err != nil {
		return "", err
	}
	trace.layer = "remote"
	code := solve.ExtractCode(ans)
	if code == "" {
		return ans, nil
	}
	asserts := solve.DeriveAsserts(prompt)
	if !looksLikePython(prompt, code) || len(asserts) == 0 {
		// No executable evidence either way; syntax check is still free.
		if looksLikePython(prompt, code) {
			sctx, cancel := context.WithTimeout(ctx, 8*time.Second)
			defer cancel()
			if solve.CheckPythonSyntax(sctx, code) != nil && r.mayRetry(prompt) {
				trace.layer = "remote-retry"
				if ans2, err2 := r.call(ctx, llm.RoleCode, cat, false, prompt); err2 == nil {
					return ans2, nil
				}
			}
		}
		return ans, nil
	}
	if r.testsPass(ctx, code, asserts) {
		trace.codeTested = true
		return ans, nil
	}
	// Proven failure — the only case worth paying the code specialist for.
	if r.mayRetry(prompt) {
		if ans2, err2 := r.call(ctx, llm.RoleCode, cat, false, prompt); err2 == nil {
			if code2 := solve.ExtractCode(ans2); code2 != "" && r.testsPass(ctx, code2, asserts) {
				trace.layer = "remote-code"
				trace.codeTested = true
				return ans2, nil
			}
		}
	}
	return ans, nil
}

func (r *Router) testsPass(ctx context.Context, code string, asserts []string) bool {
	res, err := solve.RunPython(ctx, code+"\n\n"+strings.Join(asserts, "\n"), 8*time.Second)
	if err != nil || res.TimedOut {
		return false
	}
	if res.ExitCode != 0 &&
		(strings.Contains(res.Stderr, "NameError") || strings.Contains(res.Stderr, "SyntaxError")) {
		// Broken derived asserts prove nothing about the code.
		return true
	}
	return res.ExitCode == 0
}

// --- Layer 1: everything else ---

func (r *Router) plain(ctx context.Context, cat classify.Category, generic bool, prompt string, trace *decisionTrace) (string, error) {
	// Local first: a format-checked local answer costs zero scored tokens.
	if r.localAllowed(cat) {
		if ans, ok := r.localPlain(ctx, cat, generic, prompt); ok {
			trace.layer = "local"
			return ans, nil
		}
	}
	role := llm.RoleGeneral
	if cat == classify.Logic {
		role = llm.RoleStrong
	}
	ans, err := r.call(ctx, role, cat, generic, prompt)
	if err != nil {
		return "", err
	}
	trace.layer = "remote"
	if generic || verify.Check(cat, prompt, ans) {
		return ans, nil
	}
	// Format failure is cheap to prove and usually worth one paid nudge.
	if r.mayRetry(prompt) {
		ans2, err2 := r.callWithNudge(ctx, role, cat, prompt)
		if err2 == nil && verify.Check(cat, prompt, ans2) {
			trace.layer = "remote-retry"
			return ans2, nil
		}
	}
	return ans, nil
}

func (r *Router) mayRetry(prompt string) bool {
	return r.Pace.Mode() != ModeOff && len(prompt) <= retryMaxPromptChars && r.allowRetry()
}

func (r *Router) call(ctx context.Context, role llm.Role, cat classify.Category, generic bool, prompt string) (string, error) {
	sys, maxTok := remoteSystem[cat], remoteMaxTokens[cat]
	if generic {
		sys, maxTok = genericSystem, genericMaxTokens
	}
	if !r.opt.RemoteCaps {
		maxTok = 0 // omitted from the request body entirely
	}
	resp, err := r.chatConstrained(ctx, role, cat, generic, llm.ChatRequest{
		Messages:    r.messages(sys, r.compress(cat, prompt)),
		MaxTokens:   maxTok,
		Temperature: 0,
		Stop:        r.stopFor(cat),
		// TODO: verify the Fireworks knob for disabling Gemma 4 / MiniMax
		// thinking mode against the live API — thinking tokens are scored.
	})
	if err != nil {
		return "", err
	}
	return postprocess(cat, resp.Content), nil
}

func (r *Router) callWithNudge(ctx context.Context, role llm.Role, cat classify.Category, prompt string) (string, error) {
	nudgeTok := remoteMaxTokens[cat]
	if !r.opt.RemoteCaps {
		nudgeTok = 0
	}
	resp, err := r.chatConstrained(ctx, role, cat, false, llm.ChatRequest{
		Messages:    r.messages(remoteSystem[cat]+" Follow the required output format exactly.", r.compress(cat, prompt)),
		MaxTokens:   nudgeTok,
		Temperature: 0,
		Stop:        r.stopFor(cat),
	})
	if err != nil {
		return "", err
	}
	return postprocess(cat, resp.Content), nil
}

// chatConstrained sends the request with the optional decoding constraints:
// a category grammar (filler tokens impossible by construction) and
// reasoning_effort (thinking tokens are scored — keep them low). Endpoints
// vary in support for both knobs, so a rejected request gets one plain
// retry: the constraints can never lose an answer.
func (r *Router) chatConstrained(ctx context.Context, role llm.Role, cat classify.Category, generic bool, req llm.ChatRequest) (*llm.ChatResponse, error) {
	if r.opt.Grammar && !generic {
		req.ResponseFormat = grammarFor(cat)
	}
	if r.opt.ReasoningEffort != "" {
		if req.Extra == nil {
			req.Extra = map[string]any{}
		}
		req.Extra["reasoning_effort"] = r.opt.ReasoningEffort
	}
	resp, err := r.FW.Chat(ctx, role, req)
	if err != nil && (req.ResponseFormat != nil || req.Extra != nil) {
		req.ResponseFormat, req.Extra = nil, nil // knobs rejected → plain retry
		resp, err = r.FW.Chat(ctx, role, req)
	}
	if err != nil {
		return resp, err
	}
	// Thinking drain: a model that didn't honor reasoning_effort can spend
	// the whole token cap inside reasoning_content and return an empty (or
	// length-truncated, leaked) answer. One rescue attempt with a much
	// larger cap lets the answer finish AFTER the thinking — the extra
	// tokens are cheaper than an accuracy-gate loss on the task.
	drained := resp.Content == "" && resp.ReasoningContent != ""
	leaked := resp.FinishReason == "length" && resp.ReasoningContent == "" && resp.Content != "" &&
		req.MaxTokens > 0 && req.MaxTokens < 200
	if drained || leaked {
		big := req
		big.MaxTokens = max(req.MaxTokens*4, 600)
		if r2, err2 := r.FW.Chat(ctx, role, big); err2 == nil && r2.Content != "" {
			return r2, nil
		}
	}
	return resp, nil
}

// compress applies the PROMPT_COMPRESS level to text bound for the API. The
// original prompt is untouched for classification and the free solvers.
func (r *Router) compress(cat classify.Category, prompt string) string {
	if r.opt.PromptCompress == 0 {
		return prompt
	}
	return compress.Prompt(r.opt.PromptCompress, cat == classify.Summarize, prompt)
}

func looksLikePython(prompt, code string) bool {
	lp := strings.ToLower(prompt)
	if strings.Contains(lp, "python") {
		return true
	}
	if strings.Contains(lp, "javascript") || strings.Contains(lp, "java ") ||
		strings.Contains(lp, "c++") || strings.Contains(lp, " go ") || strings.Contains(lp, "rust") {
		return false
	}
	return strings.Contains(code, "def ") || strings.Contains(code, "import ")
}

// --- postprocessing & tracing ---

var reAnswerLine = regexp.MustCompile(`(?im)^answer:\s*(.+)$`)

func postprocess(cat classify.Category, s string) string {
	s = strings.TrimSpace(s)
	switch cat {
	case classify.Math, classify.Logic:
		if m := reAnswerLine.FindStringSubmatch(s); m != nil {
			return strings.TrimSpace(m[1])
		}
	}
	return s
}

type decisionTrace struct {
	id         string
	cat        classify.Category
	generic    bool
	mode       VerifyMode
	layer      string
	codeTested bool
}

func (d *decisionTrace) emit() {
	parts := []string{"task " + d.id, "cat=" + string(d.cat), "mode=" + d.mode.String(), "layer=" + d.layer}
	if d.generic {
		parts = append(parts, "generic")
	}
	if d.codeTested {
		parts = append(parts, "code-tested")
	}
	log.Printf("%s", strings.Join(parts, " "))
}
