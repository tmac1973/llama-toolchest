package api

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestParameterControlsJS runs the benchmark job form's parameter-control
// JavaScript against a minimal DOM stub.
//
// That code was written and shipped without ever executing, and a review
// found real bugs in it — most sharply, parseFloat-based matching that
// made the dual-GPU "0-1" assignment equal to "0", silently rewriting a
// job to single-GPU on edit. Nothing in the Go test suite could see it,
// because template rendering only proves the markup parses.
//
// Skipped when node isn't installed, so it never blocks a build; it
// still runs locally and anywhere node is available.
func TestParameterControlsJS(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; skipping parameter-control JS test")
	}

	src, err := os.ReadFile(filepath.Join("..", "..", "web", "templates", "benchmarks.html"))
	if err != nil {
		t.Fatalf("read template: %v", err)
	}

	// Pull the script block out of the template and neutralize Go
	// template actions so node can parse it.
	blocks := regexp.MustCompile(`(?s)<script>(.*?)</script>`).FindAllStringSubmatch(string(src), -1)
	if len(blocks) == 0 {
		t.Fatal("no <script> block found in benchmarks.html")
	}
	var js strings.Builder
	for _, b := range blocks {
		js.WriteString(regexp.MustCompile(`\{\{[^}]*\}\}`).ReplaceAllString(b[1], "0"))
		js.WriteString("\n")
	}

	// Keep only the parameter-control functions; the rest reaches for
	// HTMX, fetch and page globals this stub deliberately doesn't model.
	wanted := []string{
		"paramRows", "paramValues", "onParamInheritToggle", "onParamValueToggle",
		"addParamCustom", "addParamValue", "syncParamRow", "readParams",
		"numEq", "fillUbatchLadder", "updateMatrixCount", "prefillJobForm",
	}
	extracted, missing := extractFunctions(js.String(), wanted)
	if len(missing) > 0 {
		t.Fatalf("functions not found in benchmarks.html (renamed or removed?): %v", missing)
	}

	dir := t.TempDir()
	for name, content := range map[string]string{
		"params.js": extracted,
		"dom.js":    mustRead(t, filepath.Join("..", "..", "web", "jstest", "dom.js")),
		"run.js":    mustRead(t, filepath.Join("..", "..", "web", "jstest", "params_test.js")),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	cmd := exec.Command(node, filepath.Join(dir, "run.js"))
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("parameter-control JS failed:\n%s", out)
	}
	if !strings.Contains(string(out), "ALL PASS") {
		t.Fatalf("unexpected JS test output:\n%s", out)
	}
}

func mustRead(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(b)
}

// extractFunctions pulls named function declarations out of a source
// blob by brace matching. Leading indentation is allowed: a page whose
// script lives inside an IIFE (visualize.html) declares its functions
// indented, and they are just as testable.
func extractFunctions(src string, names []string) (string, []string) {
	var out strings.Builder
	var missing []string
	for _, n := range names {
		re := regexp.MustCompile(`(?m)^[ \t]*function ` + n + `\(`)
		loc := re.FindStringIndex(src)
		if loc == nil {
			missing = append(missing, n)
			continue
		}
		i := strings.Index(src[loc[0]:], "{") + loc[0]
		depth, j := 0, i
		for j < len(src) {
			switch src[j] {
			case '{':
				depth++
			case '}':
				depth--
			}
			if depth == 0 {
				break
			}
			j++
		}
		out.WriteString(src[loc[0] : j+1])
		out.WriteString("\n\n")
	}
	return out.String(), missing
}
