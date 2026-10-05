package api

import (
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
	runNodeTest(t, map[string]string{
		"models.js": templateScripts(t, "models.html"),
		"run.js":    jsTestFile(t, "models_test.js"),
	})
}
