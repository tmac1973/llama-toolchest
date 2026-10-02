package api

import (
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func (s *Server) rememberSeed(downloadID string, class models.ContextClass) {
	s.seedMu.Lock()
	defer s.seedMu.Unlock()
	if s.pendingSeeds == nil {
		s.pendingSeeds = map[string]models.ContextClass{}
	}
	s.pendingSeeds[downloadID] = class
}

func (s *Server) takeSeed(downloadID string) (models.ContextClass, bool) {
	s.seedMu.Lock()
	defer s.seedMu.Unlock()
	class, ok := s.pendingSeeds[downloadID]
	delete(s.pendingSeeds, downloadID)
	return class, ok
}

// seedFromRecommendation gives a model downloaded from a recommendation
// the settings its card showed, planned again from the downloaded file
// itself, which describes the model exactly where the feed worked from
// its header. Only the settings the plan decides are copied (context,
// KV cache, GPU layers and placement, experts in system memory, threads,
// flash attention); sampling still comes from the publisher's presets.
//
// Nothing is changed when the config is no longer the default (a backup
// restored one first), for a helper model, or when the model no longer
// fits at all.
func (s *Server) seedFromRecommendation(id string, class models.ContextClass) {
	m, err := s.registry.Get(id)
	if err != nil || m.HelperRole {
		return
	}
	cur, err := s.registry.GetConfig(id)
	if err != nil {
		return
	}
	plan := models.PlanFit(m, *cur, s.hardware(), class)
	if !plan.Fits {
		slog.Info("recommended settings not applied: the model does not fit this machine any more", "model", id)
		return
	}
	requested := models.ContextClassTokens[class]
	if class == models.ContextMax || requested == 0 || (m.ContextLength > 0 && requested > m.ContextLength) {
		requested = m.ContextLength
	}

	next := *cur
	p := plan.Config
	next.ContextSize, next.KVCacheQuant = p.ContextSize, p.KVCacheQuant
	next.GPULayers, next.CPUMoE = p.GPULayers, p.CPUMoE
	next.GPUAssign, next.TensorSplit, next.SplitMode, next.MainGPU = p.GPUAssign, p.TensorSplit, p.SplitMode, p.MainGPU
	next.Threads, next.FlashAttention = p.Threads, p.FlashAttention

	seeded, err := s.registry.SeedConfig(id, next, models.SeedNote{
		Requested: requested, Context: p.ContextSize, Notes: plan.Notes, At: time.Now(),
	})
	switch {
	case err != nil:
		slog.Warn("could not apply the recommended settings", "model", id, "error", err)
	case seeded:
		slog.Info("applied the recommended settings", "model", id, "context", p.ContextSize, "kv", p.KVCacheQuant, "cpu_moe", p.CPUMoE)
	}
}

// seededNote is the config panel's sentence about settings filled in from
// a recommendation, and its tooltip: the plan's reason for each setting.
func seededNote(n *models.SeedNote) (text, tip string) {
	if n == nil {
		return "", ""
	}
	text = fmt.Sprintf("Settings suggested for this machine when this model was downloaded from the recommendations (%s context). Autoconfigure and Autotune can refine them.", tokensLabel(n.Requested))
	if n.Context > 0 && n.Requested > 0 && n.Context < n.Requested {
		text = fmt.Sprintf("Settings suggested for this machine when this model was downloaded from the recommendations. The suggested %s context did not fit any more, so %s was set. Autoconfigure and Autotune can refine them.",
			tokensLabel(n.Requested), tokensLabel(n.Context))
	}
	var reasons []string
	for _, note := range n.Notes {
		reasons = append(reasons, note.Reason)
	}
	tip = "Why each setting was chosen:\n" + strings.Join(reasons, "\n")
	if len(reasons) == 0 {
		tip = "These settings stay as they are until you change them."
	}
	return text, tip
}
