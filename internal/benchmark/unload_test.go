package benchmark

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// unloadAllModels must read the router's {"data": [...]} list. It used to
// decode a bare array, fail, and unload nothing.
func TestUnloadAllModelsUnloadsLoadedModels(t *testing.T) {
	var unloaded atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		status := "loaded"
		if unloaded.Load() {
			status = "unloaded"
		}
		io.WriteString(w, `{"data":[{"id":"left-over","status":{"value":"`+status+`"}}],"object":"list"}`)
	})
	mux.HandleFunc("/models/unload", func(w http.ResponseWriter, r *http.Request) {
		unloaded.Store(true)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	(&Runner{}).unloadAllModels(context.Background(), srv.URL)

	if !unloaded.Load() {
		t.Fatal("the loaded model was never unloaded")
	}
}
