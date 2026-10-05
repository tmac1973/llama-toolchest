package api

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestErrorNoticeJS runs web/static/errors.js against a stub document:
// a failed htmx request or apiCall shows the server's message, a retry
// that works clears it, and polling is left out.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestErrorNoticeJS(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; skipping error notice JS test")
	}
	dir := t.TempDir()
	for name, src := range map[string]string{
		"errors.js": filepath.Join("..", "..", "web", "static", "errors.js"),
		"run.js":    filepath.Join("..", "..", "web", "jstest", "errors_test.js"),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(mustRead(t, src)), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	cmd := exec.Command(node, filepath.Join(dir, "run.js"))
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL PASS") {
		t.Fatalf("error notice JS failed:\n%s", out)
	}
}
