package api

import (
	"os"
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
	// Keep only the parameter-control functions; the rest reaches for
	// HTMX, fetch and page globals this stub deliberately doesn't model.
	wanted := []string{
		"paramRows", "paramValues", "onParamInheritToggle", "onParamValueToggle",
		"addParamCustom", "addParamValue", "syncParamRow", "readParams",
		"numEq", "fillUbatchLadder", "updateMatrixCount", "prefillJobForm",
		"updateSpecValue", "updateStartLists", "checkedModelStarts",
		"profilesChosen", "prefillStarts",
	}
	extracted, missing := extractFunctions(templateScripts(t, "benchmarks.html"), wanted)
	if len(missing) > 0 {
		t.Fatalf("functions not found in benchmarks.html (renamed or removed?): %v", missing)
	}
	runNodeTest(t, map[string]string{
		"params.js": extracted,
		"dom.js":    jsTestFile(t, "dom.js"),
		"run.js":    jsTestFile(t, "params_test.js"),
	})
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
