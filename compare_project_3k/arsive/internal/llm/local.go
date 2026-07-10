package llm

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// LocalServer manages a llama-server subprocess serving the bundled GGUF.
type LocalServer struct {
	cmd     *exec.Cmd
	baseURL string
}

type LocalOptions struct {
	Bin       string
	ModelPath string
	BaseURL   string // e.g. http://127.0.0.1:8080
	CtxSize   int
	Parallel  int
	Threads   int
	ExtraArgs []string
}

// StartLocal spawns llama-server and blocks until it reports healthy or the
// context expires. The 60s container-start budget applies here: model load
// must finish well inside it.
func StartLocal(ctx context.Context, opts LocalOptions) (*LocalServer, error) {
	port := portOf(opts.BaseURL)
	args := []string{
		"-m", opts.ModelPath,
		"--port", port,
		"--host", "127.0.0.1",
		"-c", strconv.Itoa(opts.CtxSize),
		"--parallel", strconv.Itoa(opts.Parallel),
		"--no-webui",
		// Thinking burns the whole max_tokens budget inside reasoning_content
		// and leaves content empty; we prompt explicit CoT where needed instead.
		"--reasoning", "off",
	}
	if opts.Threads > 0 {
		args = append(args, "-t", strconv.Itoa(opts.Threads))
	}
	args = append(args, opts.ExtraArgs...)
	cmd := exec.Command(opts.Bin, args...)
	cmd.Stdout = os.Stderr // keep agent stdout clean; llama logs go to stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}
	s := &LocalServer{cmd: cmd, baseURL: opts.BaseURL}
	if err := s.waitReady(ctx); err != nil {
		s.Stop()
		return nil, err
	}
	return s, nil
}

func (s *LocalServer) waitReady(ctx context.Context) error {
	client := &http.Client{Timeout: 2 * time.Second}
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("llama-server not ready: %w", ctx.Err())
		case <-tick.C:
			resp, err := client.Get(s.baseURL + "/health")
			if err == nil {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return nil
				}
			}
		}
	}
}

func (s *LocalServer) Stop() {
	if s.cmd != nil && s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
	}
}

func portOf(baseURL string) string {
	if i := strings.LastIndex(baseURL, ":"); i >= 0 {
		return strings.Trim(baseURL[i+1:], "/")
	}
	return "8080"
}
