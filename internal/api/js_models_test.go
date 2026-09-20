package api

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestModelsPageListenersJS runs the models page's script against a stub.
//
// It guards the reload that keeps a card honest: Autoconfigure and
// Autotune save a profile, and may put it into use, from panels of their
// own, and the Configure panel below them goes on showing the settings
// from before the save until the page is reloaded. The server sends a
// modelConfigStale event; everything that happens next is in this script,
// where no Go test can see it.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestModelsPageListenersJS(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; skipping models page JS test")
	}

	src, err := os.ReadFile(filepath.Join("..", "..", "web", "templates", "models.html"))
	if err != nil {
		t.Fatalf("read template: %v", err)
	}
	blocks := regexp.MustCompile(`(?s)<script>(.*?)</script>`).FindAllStringSubmatch(string(src), -1)
	if len(blocks) == 0 {
		t.Fatal("no <script> block found in models.html")
	}
	var js strings.Builder
	for _, b := range blocks {
		js.WriteString(regexp.MustCompile(`\{\{[^}]*\}\}`).ReplaceAllString(b[1], "0"))
		js.WriteString("\n")
	}

	dir := t.TempDir()
	for name, content := range map[string]string{
		"models.js": js.String(),
		"run.js":    mustRead(t, filepath.Join("..", "..", "web", "jstest", "models_test.js")),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	cmd := exec.Command(node, filepath.Join(dir, "run.js"))
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("models page JS failed:\n%s", out)
	}
	if !strings.Contains(string(out), "ALL PASS") {
		t.Fatalf("unexpected JS test output:\n%s", out)
	}
}
