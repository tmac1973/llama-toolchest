package api

import (
	"path/filepath"
	"testing"
)

// TestLogPanelJS runs web/static/log-panel.js against a stub document:
// live tail scrolls on new text, Copy falls back to a textarea and shows
// a failed copy, Clear waits for the server, and a panel added by an htmx
// swap is set up once.
//
// Skipped when node isn't installed. `make js-test` runs it.
func TestLogPanelJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"log-panel.js": mustRead(t, filepath.Join("..", "..", "web", "static", "log-panel.js")),
		"dom.js":       jsTestFile(t, "dom.js"),
		"run.js":       jsTestFile(t, "log_panel_test.js"),
	})
}
