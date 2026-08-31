package api

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestVisualizeMetricFilterJS runs the visualization page's
// point-filtering against a stub.
//
// It exists because the failure it guards is invisible from Go: a run
// that carries no memory figure reaches Plotly as `undefined`, the
// formatter throws inside a library callback, and the page renders with
// an empty chart and nothing in the console to say why. Template
// rendering tests cannot see that; only running the function can.
//
// Skipped when node isn't installed. `make js-test` covers both this and
// the job form's parameter controls.
func TestVisualizeMetricFilterJS(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; skipping visualization JS test")
	}

	src, err := os.ReadFile(filepath.Join("..", "..", "web", "templates", "visualize.html"))
	if err != nil {
		t.Fatalf("read template: %v", err)
	}
	blocks := regexp.MustCompile(`(?s)<script>(.*?)</script>`).FindAllStringSubmatch(string(src), -1)
	if len(blocks) == 0 {
		t.Fatal("no <script> block found in visualize.html")
	}
	var js strings.Builder
	for _, b := range blocks {
		js.WriteString(regexp.MustCompile(`\{\{[^}]*\}\}`).ReplaceAllString(b[1], "0"))
		js.WriteString("\n")
	}

	extracted, missing := extractFunctions(js.String(), []string{"pointsWith"})
	if len(missing) > 0 {
		t.Fatalf("functions not found in visualize.html (renamed or removed?): %v", missing)
	}
	// The function reads the page's DATA; the runner supplies its own.
	extracted = "var DATA;\n" + extracted

	dir := t.TempDir()
	for name, content := range map[string]string{
		"viz.js": extracted,
		"dom.js": mustRead(t, filepath.Join("..", "..", "web", "jstest", "dom.js")),
		"run.js": mustRead(t, filepath.Join("..", "..", "web", "jstest", "viz_test.js")),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	cmd := exec.Command(node, filepath.Join(dir, "run.js"))
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("visualization JS failed:\n%s", out)
	}
	if !strings.Contains(string(out), "ALL PASS") {
		t.Fatalf("unexpected JS test output:\n%s", out)
	}
}
