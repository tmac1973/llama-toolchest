package api

import (
	"log/slog"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// placeEmbeddingModel puts embedding model id on one GPU, the one with
// the most room on hw (see models.EmbeddingGPU), while its config has no
// GPU assignment yet. A helper model, a model that is not an embedding
// model, and a config with an assignment anyone has chosen are left as
// they are. The choice is saved, so it shows in the config form, holds
// across restarts, and is what the GPU map and the room checks read.
func (s *Server) placeEmbeddingModel(id string, hw models.Hardware) {
	m, err := s.registry.Get(id)
	if err != nil || !m.IsEmbedding() || m.HelperRole {
		return
	}
	cfg, err := s.registry.GetConfig(id)
	if err != nil || cfg.GPUAssign != "" {
		return
	}
	opt, ok := models.EmbeddingGPU(m, *cfg, hw)
	if !ok {
		return
	}
	placed, err := s.registry.PlaceOnGPU(id, opt.Value, len(hw.GPUs))
	switch {
	case err != nil:
		slog.Warn("failed to place the embedding model on one GPU", "model", id, "error", err)
	case placed:
		slog.Info("placed the embedding model on one GPU", "model", id, "gpu", opt.Value)
	}
}

// placeEmbeddingModels runs placeEmbeddingModel over every registered
// model, at startup, before the router first starts: embedding models
// downloaded before one-GPU placement existed, and files ScanModels just
// found. The reading of other programs' GPU memory is taken first, as
// the monitor's watcher would, since its first one may not be in yet.
func (s *Server) placeEmbeddingModels() {
	if s.monitor == nil {
		return
	}
	s.updateOtherVRAM(s.monitor.Current())
	hw := s.idleHardware()
	for _, m := range s.registry.List() {
		s.placeEmbeddingModel(m.ID, hw)
	}
}
