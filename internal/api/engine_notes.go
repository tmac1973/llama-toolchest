package api

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/enginenotes"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// jobRef names the job that started the router, for the events recorded
// while it holds it.
type jobRef struct{ ID, Name string }

// watchEngineNotes feeds every router log line to a tracker for the life
// of the server and records what it reports. Started once, from
// NewServer.
//
// recordOutcome runs inline, in order: it reads only memory, under short
// locks, so the reader keeps up. A subscriber that falls behind loses
// lines, and a lost exit line is a crash nobody counts.
func (s *Server) watchEngineNotes() {
	tracker := enginenotes.NewTracker()
	for line := range s.process.Subscribe() {
		out := tracker.Add(line, time.Now())
		if len(out) == 0 {
			continue
		}
		for _, o := range out {
			s.recordOutcome(o)
		}
		open := tracker.Open()
		s.notesOpen.Store(&open)
	}
}

// recordOutcome saves one outcome with what the server knows about it:
// the model, the build and GPUs it ran with, and the job, if one held
// the router. A crash or a failed load also goes to the toolchest log.
func (s *Server) recordOutcome(o enginenotes.Outcome) {
	if s.notes == nil {
		return
	}
	e := enginenotes.Event{
		Time:          o.Time,
		Kind:          o.Kind,
		RouterName:    o.RouterName,
		Port:          o.Port,
		ExitStatus:    o.ExitStatus,
		LoadSeconds:   o.LoadSeconds,
		LoadedSeconds: o.LoadedSeconds,
		RouterError:   o.RouterError,
		ErrorLines:    o.ErrorLines,
		Tail:          o.Tail,
		Build:         s.runningBuild(),
	}
	if o.RouterName != "" {
		e.ModelID, e.ModelName, e.GPUs = s.describeInstance(o.RouterName)
	}
	if job := s.jobTag(); job != nil {
		e.JobID, e.JobName = job.ID, job.Name
	}
	s.notes.Record(e)

	switch o.Kind {
	case enginenotes.KindCrash, enginenotes.KindLoadFailed:
		msg := "llama-server model crashed"
		if o.Kind == enginenotes.KindLoadFailed {
			msg = "llama-server model failed to load"
		}
		firstError := ""
		if len(o.ErrorLines) > 0 {
			firstError = o.ErrorLines[0]
		}
		slog.Warn(msg, "model", e.ModelName, "exit_status", o.ExitStatus,
			"build", e.Build, "gpus", e.GPUs, "job", e.JobName, "error", firstError)
	case enginenotes.KindRouterExit:
		slog.Warn("llama-server router exited on its own", "error", o.RouterError)
	}
}

// describeInstance names the model the router calls routerName and the
// GPUs it was launched on. A model the registry no longer has keeps the
// router's name and no GPUs.
func (s *Server) describeInstance(routerName string) (id, name, gpus string) {
	m, cfg := s.findModelByAny(routerName)
	if m == nil {
		return routerName, routerName, ""
	}
	// The config the router launched with, when there is one: the saved
	// config may have been edited since, and a job runs its own.
	if running, ok := s.runningConfigFor(m.ID); ok {
		cfg = running
	}
	if cfg != nil {
		gpus = models.GPUAssignLabel(cfg.GPUAssign)
		if cfg.SplitMode == "tensor" {
			gpus += ", tensor"
		}
	}
	return m.ID, m.PublicName(), gpus
}

// resolveRouterName is the Resolver for summaries: the registry id and
// display name for a router name, or the name itself for a model that
// is gone.
func (s *Server) resolveRouterName(routerName string) (string, string) {
	if m, _ := s.findModelByAny(routerName); m != nil {
		return m.ID, m.PublicName()
	}
	return routerName, routerName
}

// jobTag is the job holding the router, or nil when none does.
func (s *Server) jobTag() *jobRef {
	env, ok := s.jobEnv()
	if !ok || !env.routerOwnedByJob() {
		return nil
	}
	return s.notesJob.Load()
}

// noteRouterStart records a router start. A job start also remembers the
// job, so the events while it runs carry its name. It runs on the
// starting goroutine, never the log reader, so reading the job store
// here is fine.
func (s *Server) noteRouterStart(job bool, build string) {
	var ref *jobRef
	if job && s.jobs != nil {
		if j, running := s.jobs.Status(); running && j != nil {
			ref = &jobRef{ID: j.ID, Name: j.Name}
		}
	}
	s.notesJob.Store(ref)
	if s.notes == nil {
		return
	}
	e := enginenotes.Event{Kind: enginenotes.KindRouterStart, StartReason: "user", Build: build}
	if job {
		e.StartReason = "job"
	}
	if ref != nil {
		e.JobID, e.JobName = ref.ID, ref.Name
	}
	s.notes.Record(e)
}

// openInstances is what the log reader last saw loaded.
func (s *Server) openInstances() []enginenotes.OpenInstance {
	if p := s.notesOpen.Load(); p != nil {
		return *p
	}
	return nil
}

// notesPeriods are the periods a summary can cover; "all" is 0.
var notesPeriods = map[string]time.Duration{
	"24h": 24 * time.Hour,
	"7d":  7 * 24 * time.Hour,
	"30d": 30 * 24 * time.Hour,
	"all": 0,
}

// notesSummary reports a period named as in notesPeriods; ok is false for
// any other name.
func (s *Server) notesSummary(period string) (enginenotes.Summary, bool) {
	if period == "" {
		period = "24h"
	}
	window, ok := notesPeriods[period]
	if !ok {
		return enginenotes.Summary{}, false
	}
	return s.notes.Summary(window, s.openInstances(), s.resolveRouterName), true
}

// handleEngineNotes answers GET /api/service/notes?period=24h|7d|30d|all.
func (s *Server) handleEngineNotes(w http.ResponseWriter, r *http.Request) {
	if s.notes == nil {
		http.Error(w, "engine notes are not available", http.StatusServiceUnavailable)
		return
	}
	sum, ok := s.notesSummary(r.URL.Query().Get("period"))
	if !ok {
		http.Error(w, "period must be 24h, 7d, 30d or all", http.StatusBadRequest)
		return
	}
	respondJSON(w, sum)
}

// notesDeleteAges are the ages the delete control offers; "all" is 0.
var notesDeleteAges = map[string]time.Duration{
	"7d":  7 * 24 * time.Hour,
	"30d": 30 * 24 * time.Hour,
	"90d": 90 * 24 * time.Hour,
	"all": 0,
}

// handleEngineNotesDelete answers POST /api/service/notes/delete with
// older_than=7d|30d|90d|all.
func (s *Server) handleEngineNotesDelete(w http.ResponseWriter, r *http.Request) {
	if s.notes == nil {
		http.Error(w, "engine notes are not available", http.StatusServiceUnavailable)
		return
	}
	age, ok := notesDeleteAges[r.FormValue("older_than")]
	if !ok {
		http.Error(w, "older_than must be 7d, 30d, 90d or all", http.StatusBadRequest)
		return
	}
	var events, hours int
	var err error
	if age == 0 {
		events, hours, err = s.notes.DeleteAll()
	} else {
		events, hours, err = s.notes.DeleteOlderThan(time.Now().Add(-age))
	}
	if err != nil {
		// The data is gone from memory either way; only the save failed,
		// and the next one tries again.
		slog.Warn("engine notes delete not saved", "error", err)
	}
	slog.Info("engine notes deleted", "older_than", r.FormValue("older_than"), "events", events, "usage_hours", hours)
	respondJSON(w, map[string]int{"removed_events": events, "removed_hours": hours})
}

// proxyModelKey carries the registry id of the model a proxied request
// names, from the proxy handler to ModifyResponse.
type proxyModelKey struct{}

func withProxyModel(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, proxyModelKey{}, id)
}

func proxyModelFrom(ctx context.Context) string {
	id, _ := ctx.Value(proxyModelKey{}).(string)
	return id
}

// usagePaths are the request paths whose responses count as model usage,
// with or without the /v1 prefix.
var usagePaths = []string{"/completions", "/embeddings", "/rerank"}

// usageCounter returns the call that counts the response to req in
// engine notes, or nil when it does not count: not a model request, a
// model we do not know, or a job holding the router (job traffic stays
// out of usage).
func (s *Server) usageCounter(req *http.Request) func(prompt, gen int, failed bool) {
	if s.notes == nil {
		return nil
	}
	counted := false
	for _, p := range usagePaths {
		if strings.HasSuffix(req.URL.Path, p) {
			counted = true
			break
		}
	}
	if !counted {
		return nil
	}
	id := proxyModelFrom(req.Context())
	if id == "" {
		return nil
	}
	if env, ok := s.jobEnv(); ok && env.routerOwnedByJob() {
		return nil
	}
	return func(prompt, gen int, failed bool) { s.notes.AddUsage(id, prompt, gen, failed) }
}

// timingTokens reads the prompt and generated token counts from
// llama.cpp's timings object; both are 0 when it has none.
func timingTokens(t map[string]any) (prompt, gen int) {
	n := func(key string) int {
		v, _ := t[key].(float64)
		return int(v)
	}
	return n("prompt_n"), n("predicted_n")
}
