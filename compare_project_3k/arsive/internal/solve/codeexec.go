package solve

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// PythonResult reports one sandboxed execution of generated/fixed code.
type PythonResult struct {
	Stdout   string
	Stderr   string
	ExitCode int
	TimedOut bool
}

// RunPython executes a snippet with a hard timeout. The container itself is
// the sandbox; this only guards against infinite loops and runaway output.
func RunPython(ctx context.Context, code string, timeout time.Duration) (*PythonResult, error) {
	dir, err := os.MkdirTemp("", "pyexec-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	file := filepath.Join(dir, "snippet.py")
	if err := os.WriteFile(file, []byte(code), 0o644); err != nil {
		return nil, err
	}

	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, "python3", "-I", file) // -I: isolated mode
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &limitedWriter{w: &stdout, n: 64 << 10}
	cmd.Stderr = &limitedWriter{w: &stderr, n: 64 << 10}
	err = cmd.Run()

	res := &PythonResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		TimedOut: cctx.Err() == context.DeadlineExceeded,
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		res.ExitCode = exitErr.ExitCode()
	} else if err != nil && !res.TimedOut {
		return nil, fmt.Errorf("run python: %w", err)
	}
	return res, nil
}

// CheckPythonSyntax compiles without executing — a cheap first gate for
// generated code before running it.
func CheckPythonSyntax(ctx context.Context, code string) error {
	cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(cctx, "python3", "-c", "import sys; compile(sys.stdin.read(), '<snippet>', 'exec')")
	cmd.Stdin = strings.NewReader(code)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("syntax error: %s", strings.TrimSpace(stderr.String()))
	}
	return nil
}

// ExtractCode pulls the first fenced code block, or returns the whole text
// when there is no fence (models told to output only code often skip fences).
func ExtractCode(text string) string {
	if i := strings.Index(text, "```"); i >= 0 {
		rest := text[i+3:]
		// Drop an optional language tag on the fence line.
		if j := strings.IndexByte(rest, '\n'); j >= 0 {
			rest = rest[j+1:]
		}
		if k := strings.Index(rest, "```"); k >= 0 {
			return strings.TrimSpace(rest[:k])
		}
		return strings.TrimSpace(rest)
	}
	return strings.TrimSpace(text)
}

type limitedWriter struct {
	w interface{ Write([]byte) (int, error) }
	n int
}

func (lw *limitedWriter) Write(p []byte) (int, error) {
	if lw.n <= 0 {
		return len(p), nil // swallow overflow silently
	}
	if len(p) > lw.n {
		p = p[:lw.n]
	}
	n, err := lw.w.Write(p)
	lw.n -= n
	return len(p), err
}
