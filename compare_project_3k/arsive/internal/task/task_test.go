package task

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The judge matches results on the echoed task_id and reads /output as a
// (possibly non-root) separate process: ids must round-trip byte-for-byte
// whatever their JSON type, and the file must be world-readable.
func TestReadWriteRoundTrip(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "tasks.json")
	out := filepath.Join(dir, "results.json")
	input := `[{"task_id": 7, "prompt": "2+2?"}, {"task_id": "t2", "prompt": "hi"}, {"prompt": "no id"}]`
	if err := os.WriteFile(in, []byte(input), 0o644); err != nil {
		t.Fatal(err)
	}

	tasks, err := Read(in)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if len(tasks) != 3 {
		t.Fatalf("got %d tasks, want 3", len(tasks))
	}
	if tasks[0].ID != "7" || tasks[1].ID != "t2" || tasks[2].ID != "idx_2" {
		t.Fatalf("internal ids = %q, %q, %q", tasks[0].ID, tasks[1].ID, tasks[2].ID)
	}

	results := make([]Result, len(tasks))
	for i, tk := range tasks {
		results[i] = Result{ID: tk.ID, RawID: tk.RawID, Answer: "x"}
	}
	if err := WriteAtomic(out, results); err != nil {
		t.Fatalf("WriteAtomic: %v", err)
	}

	data, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var echoed []struct {
		ID     json.RawMessage `json:"task_id"`
		Answer string          `json:"answer"`
	}
	if err := json.Unmarshal(data, &echoed); err != nil {
		t.Fatalf("output not valid JSON: %v", err)
	}
	// Numeric id stays numeric, string stays string, missing id is fabricated.
	if got := string(echoed[0].ID); got != "7" {
		t.Errorf("task 0 id echoed as %s, want 7", got)
	}
	if got := string(echoed[1].ID); got != `"t2"` {
		t.Errorf(`task 1 id echoed as %s, want "t2"`, got)
	}
	if got := string(echoed[2].ID); got != `"idx_2"` {
		t.Errorf(`task 2 id echoed as %s, want "idx_2"`, got)
	}

	fi, err := os.Stat(out)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode().Perm() != 0o644 {
		t.Errorf("results.json mode = %o, want 644 (judge may read as non-root)", fi.Mode().Perm())
	}
}
