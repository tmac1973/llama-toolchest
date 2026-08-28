package backup

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Parse is the structural gate: shape errors reject the whole file;
// value errors (bad preset names, unknown env vars, unparseable GPU
// assigns, missing paths) are per-item concerns handled during Apply.
// Shape errors are exactly: invalid JSON, an unsupported version, a
// model config entry with an empty model_id/quant/filename, or a flag
// preset with an empty name/profile. All shape errors are collected into
// one rejection; nothing is applied if Parse fails.
func Parse(data []byte) (*File, error) {
	var f File
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, fmt.Errorf("not a valid backup file: %w", err)
	}
	if f.Version != Version {
		return nil, fmt.Errorf("backup version %d is not supported by this build (want %d)", f.Version, Version)
	}
	var shape []string
	for i, mc := range f.ModelConfigs {
		if mc.ModelID == "" || mc.Quant == "" || mc.Filename == "" {
			shape = append(shape, fmt.Sprintf("model_configs[%d]: model_id, quant, and filename are required", i))
		}
	}
	for i, p := range f.FlagPresets {
		if p.Name == "" || p.Profile == "" {
			shape = append(shape, fmt.Sprintf("flag_presets[%d]: name and profile are required", i))
		}
	}
	if len(shape) > 0 {
		return nil, fmt.Errorf("malformed backup file: %s", strings.Join(shape, "; "))
	}
	return &f, nil
}

// Selections chooses which sections a restore applies. Server-side and
// authoritative regardless of what the client preview showed.
type Selections struct {
	Settings     bool
	RuntimeEnv   bool
	FlagPresets  bool
	ModelConfigs bool
}

// None reports an empty selection.
func (s Selections) None() bool {
	return !s.Settings && !s.RuntimeEnv && !s.FlagPresets && !s.ModelConfigs
}

// SkippedItem is one item that failed to apply, with the reason.
type SkippedItem struct {
	Item   string `json:"item"`
	Reason string `json:"reason"`
}

// MissingModel is a config whose model isn't installed on this machine.
// Config carries the topology-normalized launch config (path fields
// still in exported form — existence is only checkable once the model's
// files arrive); Pending reports whether it was held for auto-claim.
type MissingModel struct {
	ModelID  string             `json:"model_id"`
	Quant    string             `json:"quant"`
	Filename string             `json:"filename"`
	Config   models.ModelConfig `json:"config"`
	Pending  bool               `json:"pending"`
}

// Report itemizes what a restore did. Error is set only for whole-file
// refusals (structural rejection, busy, empty selection) rendered
// through the same partial so failures are always visible.
type Report struct {
	Error               string         `json:"error,omitempty"`
	Applied             []string       `json:"applied,omitempty"`
	AppliedModelConfigs int            `json:"applied_model_configs"`
	Notes               []string       `json:"notes,omitempty"`
	Warnings            []string       `json:"warnings,omitempty"`
	Skipped             []SkippedItem  `json:"skipped,omitempty"`
	NotSelected         []string       `json:"not_selected,omitempty"`
	Missing             []MissingModel `json:"missing,omitempty"`
}

// Deps injects the collaborators Apply writes through, keeping the
// engine testable without the api package. Every mutation goes through
// an existing single-source setter on the api side.
type Deps struct {
	// ApplySettings merges the non-nil preference fields and reports
	// which changed value.
	ApplySettings func(Settings) (changed []string, err error)
	// CurrentEnv returns the target's live runtime env — the engine
	// needs it to build the never-deletes merge.
	CurrentEnv func() RuntimeEnv
	// ApplyEnv stores the engine-built, engine-validated merged set.
	ApplyEnv       func(RuntimeEnv) error
	SaveFlagPreset func(builder.FlagPreset) error
	// InstalledModels returns every registry ID matching the identity;
	// empty means not installed. Plural: duplicate registrations of the
	// same quant exist in practice and the config applies to every one.
	InstalledModels  func(modelID, quant string) []string
	ApplyModelConfig func(registryID string, cfg models.ModelConfig) error
	// SavePending holds a missing model's config for auto-claim. Nil
	// disables pending (the entry stays a skip).
	SavePending func(MissingModel) error
	NumGPUs     int
	ModelsDir   string
}

// Apply runs the selected sections item by item. It never deletes:
// settings merge field-wise, env merges per key, presets and configs
// upsert.
func Apply(f *File, sel Selections, deps Deps) Report {
	var rep Report

	note := func(section string, present bool) bool {
		if !present {
			return false
		}
		rep.NotSelected = append(rep.NotSelected, section)
		return true
	}

	restartReminder := false

	// Settings.
	if sel.Settings && f.Settings != nil {
		s := *f.Settings
		// Defensive: a present-but-empty secret must never blank the
		// target's credential. Assemble never emits these; hand-edited
		// files might.
		if s.HFToken != nil && *s.HFToken == "" {
			s.HFToken = nil
			rep.Warnings = append(rep.Warnings, "settings: empty hf_token ignored")
		}
		if s.MSToken != nil && *s.MSToken == "" {
			s.MSToken = nil
			rep.Warnings = append(rep.Warnings, "settings: empty ms_token ignored")
		}
		if s.APIKey != nil && *s.APIKey == "" {
			s.APIKey = nil
			rep.Warnings = append(rep.Warnings, "settings: empty api_key ignored")
		}
		changed, err := deps.ApplySettings(s)
		if err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"settings", err.Error()})
		} else {
			if len(changed) > 0 {
				rep.Applied = append(rep.Applied, "settings: "+strings.Join(changed, ", "))
				restartReminder = true
			} else {
				rep.Applied = append(rep.Applied, "settings: no changes")
			}
		}
	} else if !sel.Settings {
		note("settings", f.Settings != nil)
	}

	// Runtime env: per-key merge honoring never-deletes.
	if sel.RuntimeEnv && f.RuntimeEnv != nil {
		cur := deps.CurrentEnv()
		merged := RuntimeEnv{Curated: map[string]string{}, Extra: cur.Extra}
		for k, v := range cur.Curated {
			merged.Curated[k] = v
		}
		for k, v := range f.RuntimeEnv.Curated {
			merged.Curated[k] = v
		}
		if strings.TrimSpace(f.RuntimeEnv.Extra) != "" {
			merged.Extra = f.RuntimeEnv.Extra
		}
		set := config.EnvSet{Curated: merged.Curated, Extra: merged.Extra}
		if err := set.Validate(); err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"runtime env", err.Error()})
		} else if err := deps.ApplyEnv(merged); err != nil {
			rep.Skipped = append(rep.Skipped, SkippedItem{"runtime env", err.Error()})
		} else {
			rep.Applied = append(rep.Applied, fmt.Sprintf("runtime env: %d variables", len(merged.Curated)))
			rep.Warnings = append(rep.Warnings, set.Warnings()...)
			restartReminder = true
		}
	} else if !sel.RuntimeEnv {
		note("runtime env", f.RuntimeEnv != nil)
	}

	// Flag presets: upsert each.
	if sel.FlagPresets {
		for _, p := range f.FlagPresets {
			if err := deps.SaveFlagPreset(p); err != nil {
				rep.Skipped = append(rep.Skipped, SkippedItem{"flag preset " + p.Name, err.Error()})
				continue
			}
			rep.Applied = append(rep.Applied, "flag preset: "+p.Name)
		}
	} else {
		note("flag presets", len(f.FlagPresets) > 0)
	}

	// Model configs: normalize-then-match, so Missing entries carry a
	// config already valid for this machine and pending claim is a plain
	// attach.
	if sel.ModelConfigs {
		for _, mc := range f.ModelConfigs {
			ident := mc.ModelID + " " + mc.Quant
			cfg := mc.Config

			// Topology normalization first.
			if a := cfg.GPUAssign; a != "" && a != "custom" {
				switch {
				case deps.NumGPUs == 0:
					rep.Warnings = append(rep.Warnings, ident+": no GPUs detected — GPU assignment imported verbatim")
				case models.AssignGPUsOutOfRange(a, deps.NumGPUs):
					cfg.GPUAssign = "all"
					cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU = models.ResolveGPUAssign("all", deps.NumGPUs)
					rep.Warnings = append(rep.Warnings, fmt.Sprintf(
						"%s: GPU assignment %q references GPUs this machine doesn't have — reset to all GPUs", ident, a))
				default:
					cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU = models.ResolveGPUAssign(a, deps.NumGPUs)
				}
			} else if a == "custom" {
				rep.Warnings = append(rep.Warnings, ident+": custom tensor split imported verbatim — verify it against this machine's GPUs")
			}

			ids := deps.InstalledModels(mc.ModelID, mc.Quant)
			if len(ids) == 0 {
				missing := MissingModel{ModelID: mc.ModelID, Quant: mc.Quant, Filename: mc.Filename, Config: cfg}
				if deps.SavePending != nil {
					if err := deps.SavePending(missing); err != nil {
						rep.Skipped = append(rep.Skipped, SkippedItem{ident, "pending save failed: " + err.Error()})
					} else {
						missing.Pending = true
					}
				} else {
					rep.Skipped = append(rep.Skipped, SkippedItem{ident, "model not installed"})
				}
				rep.Missing = append(rep.Missing, missing)
				continue
			}

			// Matched: paths are resolvable now.
			rep.Warnings = append(rep.Warnings, prefixAll(ident, models.ResolveConfigPaths(&cfg, deps.ModelsDir))...)
			for _, id := range ids {
				if err := deps.ApplyModelConfig(id, cfg); err != nil {
					rep.Skipped = append(rep.Skipped, SkippedItem{ident, err.Error()})
					continue
				}
				rep.AppliedModelConfigs++
				rep.Applied = append(rep.Applied, "model config: "+ident)
			}
		}
	} else {
		note("model configs", len(f.ModelConfigs) > 0)
	}

	if restartReminder {
		rep.Notes = append(rep.Notes, "these changes take effect on the next server restart")
	}
	return rep
}

func prefixAll(prefix string, msgs []string) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = prefix + ": " + m
	}
	return out
}
