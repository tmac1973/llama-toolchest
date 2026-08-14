package api

import (
	"context"
	"log/slog"
	"time"
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
		return // sampling presets are meaningless for embedding models
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
