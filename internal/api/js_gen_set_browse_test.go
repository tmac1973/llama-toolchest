package api

import (
	"path/filepath"
	"testing"
)

// The Browse dialog's Use button: see web/jstest/gen_set_browse_test.js.
func TestGenSetBrowseJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"gen-set-browse.js": mustRead(t, filepath.Join("..", "..", "web", "static", "gen-set-browse.js")),
		"run.js":            jsTestFile(t, "gen_set_browse_test.js"),
	})
}
