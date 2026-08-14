package models

// SamplingPreset is a named bundle of sampling parameters recommended by the
// model author or quant publisher. Pointer fields stay nil when the source
// didn't publish a value, so the UI can leave the corresponding input blank
// (i.e. "use server default") rather than forcing a number.
type SamplingPreset struct {
	Name            string   `json:"name"`                       // stable id within a repo, e.g. "thinking"
	Label           string   `json:"label"`                      // human-friendly display name
	Description     string   `json:"description,omitempty"`      // short context line shown next to the dropdown
	Source          string   `json:"source"`                     // "generation_config.json" or "readme"
	SourceURL       string   `json:"source_url,omitempty"`       // link the user can open to verify
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`
	MinP            *float64 `json:"min_p,omitempty"`
	PresencePenalty *float64 `json:"presence_penalty,omitempty"`
	RepeatPenalty   *float64 `json:"repeat_penalty,omitempty"`
}

// LookupSamplingPresets returns the preset variants known for a HuggingFace
// repo ID. The lookup is exact-match on the repo ID stored in Model.ModelID
// (e.g. "unsloth/Qwen3-32B-GGUF"). Returns nil for unknown repos.
//
// Data is generated at scrape time by scripts/scrape-sampling-presets and
// lives in sampling_presets_data.go. To refresh: make scrape-sampling-presets.
func LookupSamplingPresets(repoID string) []SamplingPreset {
	return samplingPresetsByRepo[repoID]
}

// EffectiveSamplingPresets returns the presets attached to this model record
// (gathered at download/scan time), falling back to the legacy compiled-in
// scrape snapshot for records that predate download-time discovery.
func (m *Model) EffectiveSamplingPresets() []SamplingPreset {
	if len(m.SamplingPresets) > 0 {
		return m.SamplingPresets
	}
	return LookupSamplingPresets(m.ModelID)
}

// UpsertSamplingPreset returns presets with p replacing the entry of the same
// Name, inserting it otherwise ("default" first, others appended). Preset
// sources upsert by variant name so the list keeps one entry per variant.
func UpsertSamplingPreset(presets []SamplingPreset, p SamplingPreset) []SamplingPreset {
	for i := range presets {
		if presets[i].Name == p.Name {
			presets[i] = p
			return presets
		}
	}
	if p.Name == "default" {
		return append([]SamplingPreset{p}, presets...)
	}
	return append(presets, p)
}

// SamplingPresetsCoverage reports how many repos have presets registered.
// Used by tests and the scraper's self-check.
func SamplingPresetsCoverage() int {
	return len(samplingPresetsByRepo)
}

// fptr / iptr are used by the generated data file. They're exported as
// lowercase helpers so the generated source stays compact.
func fptr(v float64) *float64 { return &v }
func iptr(v int) *int         { return &v }
