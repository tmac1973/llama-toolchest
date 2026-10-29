package api

import "testing"

// TestMeasureNeedsJS runs the Measure dialog's genSizeNeedText from
// server.html: what it says beside each video size.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestMeasureNeedsJS(t *testing.T) {
	extracted, missing := extractFunctions(templateScripts(t, "server.html"), []string{"genSizeNeedText"})
	if len(missing) > 0 {
		t.Fatalf("functions not found in server.html (renamed or removed?): %v", missing)
	}
	runNodeTest(t, map[string]string{
		"measure.js": extracted,
		"run.js":     jsTestFile(t, "measure_needs_test.js"),
	})
}
