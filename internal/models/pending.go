package models

import (
	"log/slog"
	"sort"
	"time"
)

// PendingConfig is a model launch config imported from a backup for a
// model that isn't installed yet, held for auto-claim: when a model with
// matching (ModelID, Quant) registers — downloaded through any path or
// found by a scan — the config attaches automatically. The stored
// config is already topology-normalized for this machine (the restore
// engine normalizes before saving pending); path fields stay in
// exported form until claim, when their existence becomes checkable.
type PendingConfig struct {
	ModelID  string      `json:"model_id"`
	Quant    string      `json:"quant"`
	Filename string      `json:"filename"` // required — drives the download offer
	Config   ModelConfig `json:"config"`
	SavedAt  time.Time   `json:"saved_at"`
}

// SetPendingConfig upserts a pending entry by identity — re-importing a
// backup refreshes it rather than duplicating.
func (r *Registry) SetPendingConfig(p PendingConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range r.data.PendingConfigs {
		if r.data.PendingConfigs[i].ModelID == p.ModelID && r.data.PendingConfigs[i].Quant == p.Quant {
			r.data.PendingConfigs[i] = p
			r.save()
			return nil
		}
	}
	r.data.PendingConfigs = append(r.data.PendingConfigs, p)
	sort.Slice(r.data.PendingConfigs, func(i, j int) bool {
		a, b := r.data.PendingConfigs[i], r.data.PendingConfigs[j]
		if a.ModelID != b.ModelID {
			return a.ModelID < b.ModelID
		}
		return a.Quant < b.Quant
	})
	r.save()
	return nil
}

// PendingConfigs returns a copy of the pending entries, sorted.
func (r *Registry) PendingConfigs() []PendingConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]PendingConfig, len(r.data.PendingConfigs))
	copy(out, r.data.PendingConfigs)
	return out
}

// DiscardPendingConfig removes a pending entry, reporting whether it
// existed.
func (r *Registry) DiscardPendingConfig(modelID, quant string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, p := range r.data.PendingConfigs {
		if p.ModelID == modelID && p.Quant == quant {
			r.data.PendingConfigs = append(r.data.PendingConfigs[:i], r.data.PendingConfigs[i+1:]...)
			r.save()
			return true
		}
	}
	return false
}

// claimPendingLocked attaches a pending config to a just-registered
// model when identities match. Called from Add (which both downloads and
// ScanModels route through) with r.mu held.
//
// GPU placement is deliberately not re-resolved here: the stored config
// was normalized for this machine at import time, and topology changes
// between import and claim have the same (unguarded) exposure as any
// saved config after a hardware change. Path fields are resolved now —
// the model's files, including any sibling mmproj, have just arrived.
// Dirty-marking is unnecessary: a just-arrived model has never been
// loaded, so there is no running config to diverge from; the preset
// picks the config up on the next regeneration.
func (r *Registry) claimPendingLocked(m *Model) {
	for i, p := range r.data.PendingConfigs {
		if p.ModelID != m.ModelID || p.Quant != m.Quant {
			continue
		}
		cfg := p.Config
		for _, w := range ResolveConfigPaths(&cfg, r.modelsDir) {
			slog.Warn("pending config path", "model", m.ID, "warning", w)
		}
		r.data.Configs[m.ID] = &cfg
		r.data.PendingConfigs = append(r.data.PendingConfigs[:i], r.data.PendingConfigs[i+1:]...)
		slog.Info("pending config claimed", "model", m.ID, "model_id", p.ModelID, "quant", p.Quant)
		return
	}
}
