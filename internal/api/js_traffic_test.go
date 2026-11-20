package api

import (
	"path/filepath"
	"testing"
)

// TestTrafficJS runs the store of web/static/traffic.js (the live traffic
// viewer): parallel streams stay apart, the model and engine filters,
// the request limit, and answers put back together from SSE pieces.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestTrafficJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"traffic.js": mustRead(t, filepath.Join("..", "..", "web", "static", "traffic.js")),
		"run.js":     jsTestFile(t, "traffic_test.js"),
	})
}
