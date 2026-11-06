package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// A status reply that is not JSON ends the job with an error and cancels
// it on ace-server. Polling on would never see "done" and would run until
// the caller's context ended.
func TestRunAceJobStopsOnUnreadableStatus(t *testing.T) {
	var cancelled atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/synth":
			w.Write([]byte(`{"id":"j1"}`))
		case r.URL.Path == "/job" && r.URL.Query().Get("cancel") == "1":
			cancelled.Store(true)
		case r.URL.Path == "/job":
			w.Write([]byte(`<html>busy</html>`))
		}
	}))
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, _, err := runAceJob(ctx, srv.URL, "/synth", map[string]any{}, nil)
	if err == nil {
		t.Fatal("runAceJob returned no error for an unreadable status")
	}
	if ctx.Err() != nil {
		t.Fatal("runAceJob polled until the context ended")
	}
	if !cancelled.Load() {
		t.Error("the job was not cancelled on ace-server")
	}
}
