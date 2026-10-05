package api

import (
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
	extracted, missing := extractFunctions(templateScripts(t, "visualize.html"), []string{"pointsWith"})
	if len(missing) > 0 {
		t.Fatalf("functions not found in visualize.html (renamed or removed?): %v", missing)
	}
	runNodeTest(t, map[string]string{
		// The function reads the page's DATA; the runner supplies its own.
		"viz.js": "var DATA;\n" + extracted,
		"dom.js": jsTestFile(t, "dom.js"),
		"run.js": jsTestFile(t, "viz_test.js"),
	})
}
