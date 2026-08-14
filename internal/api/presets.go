package api

import (
	"context"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
)

// enrichModelPresets runs the network preset source chain for one model and
// persists the result. Called in a goroutine from the download-completion
// hook and the startup backfill — it must never block or fail a download, so
// everything here is best-effort with a hard timeout. The attempt time is
// stamped even on a total miss, so repos that legitimately publish nothing
// aren't re-fetched every boot.
func (s *Server) enrichModelPresets(id string) {
	m, err := s.registry.Get(id)
	if err != nil {
		return
	}
	if m.IsEmbedding() {
		// Sampling presets are meaningless for embedding models — just stamp
		// the attempt so backfill doesn't reconsider them every boot.
		s.registry.SetSamplingPresets(id, m.SamplingPresets, time.Now())
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	before := len(m.SamplingPresets)
	merged := s.presets.Fetch(ctx, m)
	if err := s.registry.SetSamplingPresets(id, merged, time.Now()); err != nil {
		return // model deleted while we were fetching
	}
	if len(merged) > before {
		slog.Info("fetched sampling presets", "model", id, "presets", len(merged))
	}
}

// backfillPresets runs the network preset chain once for every model that has
// never been attempted — models downloaded before this feature existed and
// files found by ScanModels. Serial with jittered pacing out of politeness to
// the upstream services; a miss stamps PresetsCheckedAt, so worst case the
// registry converges over a few boots rather than hammering anyone.
func (s *Server) backfillPresets() {
	ids := s.registry.ListNeedingPresetFetch()
	if len(ids) == 0 {
		return
	}
	slog.Info("backfilling sampling presets", "models", len(ids))
	for i, id := range ids {
		if i > 0 {
			time.Sleep(time.Second + time.Duration(rand.IntN(2000))*time.Millisecond)
		}
		s.enrichModelPresets(id)
	}
}

// handleRefreshPresets re-runs the preset source chain for one model on
// demand — e.g. after the publisher adds a docs page for a model grabbed on
// release day. Synchronous so the re-rendered config form shows the result.
func (s *Server) handleRefreshPresets(w http.ResponseWriter, r *http.Request) {
	id := s.registry.ResolveID(chi.URLParam(r, "id"))
	if _, err := s.registry.Get(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	s.enrichModelPresets(id)
	if isHTMX(r) {
		s.handleGetModelConfig(w, r)
		return
	}
	m, err := s.registry.Get(id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	respondJSON(w, map[string]any{"presets": m.EffectiveSamplingPresets()})
}
