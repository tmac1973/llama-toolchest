package api

import (
	"path/filepath"
	"testing"
)

// TestKeepInViewJS runs web/static/keep-in-view.js against a stub
// document: a click that swaps a panel for a shorter one above the screen
// scrolls it back, and polling or a panel still on screen does not.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestKeepInViewJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"keep-in-view.js": mustRead(t, filepath.Join("..", "..", "web", "static", "keep-in-view.js")),
		"dom.js":          jsTestFile(t, "dom.js"),
		"run.js":          jsTestFile(t, "keep_in_view_test.js"),
	})
}
