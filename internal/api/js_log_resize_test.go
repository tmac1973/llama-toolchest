package api

import (
	"path/filepath"
	"testing"
)

// TestLogResizeJS runs web/jstest/log_resize_test.js against
// web/static/log-resize.js: the Server log's grip sets, saves, clamps and
// resets the log height.
func TestLogResizeJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"log-resize.js": mustRead(t, filepath.Join("..", "..", "web", "static", "log-resize.js")),
		"run.js":        jsTestFile(t, "log_resize_test.js"),
	})
}
