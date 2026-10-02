package models

import "time"

// SeedNote records that a model's config was filled in from a
// recommendation when it finished downloading.
type SeedNote struct {
	// Requested is the context the recommendation was for; Context is
	// what the plan gave on this machine, which can be less when the
	// hardware changed in between.
	Requested int `json:"requested"`
	Context   int `json:"context"`
	// Notes are the plan's reasons for each setting.
	Notes []ProfileNote `json:"notes,omitempty"`
	At    time.Time     `json:"at"`
}

// SeedConfig replaces a model's config with cfg and marks it as seeded,
// but only while the config is still the one a new model starts with
// (DefaultConfig). A config restored from a backup, or one already
// changed, is left alone and SeedConfig returns false.
func (r *Registry) SeedConfig(id string, cfg ModelConfig, note SeedNote) (bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	m, cur, err := r.modelAndConfigLocked(id)
	if err != nil {
		return false, err
	}
	if !ProfileEqual(*cur, DefaultConfig()) || cur.Enabled != DefaultConfig().Enabled || cur.ActiveProfile != "" {
		return false, nil
	}
	cfg.Enabled, cfg.Aliases = cur.Enabled, cur.Aliases
	*cur = cloneConfig(cfg)
	m.Seeded = &note
	return true, r.save()
}

// ClearSeeded removes the seeded mark: the config is now someone's own.
func (r *Registry) ClearSeeded(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.data.Models[id]
	if !ok || m.Seeded == nil {
		return nil
	}
	if err := r.writableLocked(); err != nil {
		return err
	}
	m.Seeded = nil
	return r.save()
}
