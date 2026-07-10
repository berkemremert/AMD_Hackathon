package task

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Task struct {
	ID     string          // string form for internal use (logs, lookups)
	RawID  json.RawMessage // task_id exactly as it appeared in the input
	Prompt string
}

// UnmarshalJSON tolerates any JSON scalar as task_id (the judge matches on
// the echoed value, so a numeric id must not fail the whole parse) and keeps
// the raw bytes so the output round-trips it exactly.
func (t *Task) UnmarshalJSON(b []byte) error {
	var aux struct {
		ID     json.RawMessage `json:"task_id"`
		Prompt string          `json:"prompt"`
	}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	t.RawID = aux.ID
	t.Prompt = aux.Prompt
	var s string
	if json.Unmarshal(aux.ID, &s) == nil {
		t.ID = s
	} else {
		t.ID = string(bytes.TrimSpace(aux.ID))
	}
	return nil
}

type Result struct {
	ID     string
	RawID  json.RawMessage
	Answer string
}

// MarshalJSON echoes task_id byte-for-byte from the input when available so
// the judge's id matching can never miss on type or formatting.
func (r Result) MarshalJSON() ([]byte, error) {
	id := r.RawID
	if len(id) == 0 {
		b, err := json.Marshal(r.ID)
		if err != nil {
			return nil, err
		}
		id = b
	}
	return json.Marshal(struct {
		ID     json.RawMessage `json:"task_id"`
		Answer string          `json:"answer"`
	}{id, r.Answer})
}

func Read(path string) ([]Task, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tasks: %w", err)
	}
	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("parse tasks: %w", err)
	}
	// A task with no usable id still needs a stable, unique echo value.
	for i := range tasks {
		if tasks[i].ID == "" && len(tasks[i].RawID) == 0 {
			tasks[i].ID = fmt.Sprintf("idx_%d", i)
		}
	}
	return tasks, nil
}

// WriteAtomic writes results as valid JSON via a temp file + rename, so a
// crash mid-write can never leave a malformed results.json behind.
func WriteAtomic(path string, results []Result) error {
	if results == nil {
		results = []Result{}
	}
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir output: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".results-*.json")
	if err != nil {
		return fmt.Errorf("temp file: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("write results: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("close results: %w", err)
	}
	// CreateTemp makes 0600 files and the container runs as root: a judge
	// process reading the mounted /output as a non-root user would get
	// permission denied and score the run 0%. World-readable like a plain
	// open() would produce.
	if err := os.Chmod(tmpName, 0o644); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("chmod results: %w", err)
	}
	return os.Rename(tmpName, path)
}
