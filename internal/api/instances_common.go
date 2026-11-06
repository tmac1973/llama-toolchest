package api

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/process"
)

// The generation and fork instance handlers share these helpers. Both
// managers embed process.Instances, which is what they take.

// urlInstance reads the {id} instance through get, writing a 404 when it
// is unknown.
func urlInstance[T any](w http.ResponseWriter, r *http.Request, get func(string) (T, error)) (T, bool) {
	inst, err := get(chi.URLParam(r, "id"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return inst, false
	}
	return inst, true
}

// deleteStoppedInstance deletes the instance id through del and drops its
// supervisor. It refuses with 409 while the instance is starting or
// running. It reports whether the instance was deleted; on false it has
// written the error.
func deleteStoppedInstance(w http.ResponseWriter, procs *process.Instances, id string, del func(string) error) bool {
	if procs.IsActive(id) {
		http.Error(w, "stop the instance before deleting it", http.StatusConflict)
		return false
	}
	if err := del(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return false
	}
	procs.Forget(id)
	return true
}

// streamInstanceLogs streams the {id} instance's log lines until the
// client goes away; 404 when it has not run yet.
func streamInstanceLogs(w http.ResponseWriter, r *http.Request, procs *process.Instances) {
	id := chi.URLParam(r, "id")
	ch := procs.Subscribe(id)
	if ch == nil {
		http.Error(w, "this instance has not run yet", http.StatusNotFound)
		return
	}
	defer procs.Unsubscribe(id, ch)
	StreamLines(w, r.Context(), ch, "Instance stopped")
}

// instancesChanged answers a change with list (the refreshed table for
// htmx, the instance list as JSON otherwise), and tells htmx pages to
// refresh through the event.
func instancesChanged(w http.ResponseWriter, r *http.Request, event string, list http.HandlerFunc) {
	if isHTMX(r) {
		w.Header().Set("HX-Trigger", event)
	}
	list(w, r)
}

// autoStartInstances starts each of ids through start, logging the
// outcome; kind names the instances in the log ("generation", "fork").
// With afterRouter it first waits for the router to be running or to have
// failed, for up to 5 minutes.
func (s *Server) autoStartInstances(kind string, afterRouter bool, ids []string, start func(string) error) {
	if afterRouter {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		_ = s.process.WaitRunning(ctx)
		cancel()
	}
	for _, id := range ids {
		if err := start(id); err != nil {
			slog.Warn("failed to auto-start a "+kind+" instance", "instance", id, "error", err)
		} else {
			slog.Info("auto-started "+kind+" instance", "instance", id)
		}
	}
}

// featureOff answers a request for a feature that is switched off in
// Settings: 404, with what to do.
func featureOff(w http.ResponseWriter, what string) {
	http.Error(w, what+" are switched off; switch them on in Settings", http.StatusNotFound)
}
