package api

import (
	"path/filepath"
	"testing"
)

// TestErrorNoticeJS runs web/static/errors.js against a stub document:
// a failed htmx request or apiCall shows the server's message, a retry
// that works clears it, and polling is left out.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestErrorNoticeJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"errors.js": mustRead(t, filepath.Join("..", "..", "web", "static", "errors.js")),
		"run.js":    jsTestFile(t, "errors_test.js"),
	})
}
