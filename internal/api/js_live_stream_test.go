package api

import (
	"path/filepath"
	"testing"
)

// TestLiveStreamJS runs web/jstest/live_stream_test.js against
// web/static/live-stream.js: streams close in hidden tabs and on the done
// event, so open tabs cannot use up the browser's connections.
func TestLiveStreamJS(t *testing.T) {
	runNodeTest(t, map[string]string{
		"live-stream.js": mustRead(t, filepath.Join("..", "..", "web", "static", "live-stream.js")),
		"run.js":         jsTestFile(t, "live_stream_test.js"),
	})
}
