package builder

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// FlagPreset is a saved build-flag set for a profile: toggle states plus
// extra cmake flags, named by the user via the Build Tag field. It
// deliberately excludes the git ref — the point is replaying a favorite
// flag set against new refs — and is scoped to a profile, since toggles
// only exist per backend.
type FlagPreset struct {
	Name       string          `json:"name"`
	Profile    string          `json:"profile"`
	Options    map[string]bool `json:"options"`
	ExtraCMake string          `json:"extra_cmake,omitempty"`
	// Version is the toggle meaning the preset was saved under (see
	// FlagPresetVersion). Missing (0) means saved before versions existed.
	Version int `json:"version,omitempty"`
}

// FlagPresetVersion is the toggle meaning current presets are saved
// under. Raise it when a toggle's stored true/false changes what gets
// built, and add the step to migrateFlagPreset.
//
// 1: the CUDA Graphs toggle defaults to on and writes
// GGML_CUDA_GRAPHS=OFF when off. Before, off wrote nothing and llama.cpp
// built with graphs on anyway, so both stored values meant on.
const FlagPresetVersion = 1

// migrateFlagPreset brings a preset saved under an older FlagPresetVersion
// up to the current one without changing what it builds. It reports
// whether anything changed.
func migrateFlagPreset(p *FlagPreset) bool {
	if p.Version >= FlagPresetVersion {
		return false
	}
	if p.Version < 1 && p.Profile == "cuda" {
		// Set even when absent: a loaded preset treats a missing toggle
		// as off, which would now mean OFF.
		if p.Options == nil {
			p.Options = map[string]bool{}
		}
		p.Options["GGML_CUDA_GRAPHS"] = true
	}
	p.Version = FlagPresetVersion
	return true
}

func (b *Builder) flagPresetsPath() string {
	return filepath.Join(b.dataDir, "config", "build-flag-presets.json")
}

// loadFlagPresetsLocked reads the store from disk once. Callers hold fpMu.
func (b *Builder) loadFlagPresetsLocked() {
	if b.fpLoaded {
		return
	}
	b.fpLoaded = true
	data, err := os.ReadFile(b.flagPresetsPath())
	if err != nil {
		return
	}
	json.Unmarshal(data, &b.flagPresets)
	changed := false
	for i := range b.flagPresets {
		if migrateFlagPreset(&b.flagPresets[i]) {
			changed = true
		}
	}
	if changed {
		b.saveFlagPresetsLocked()
	}
}

func (b *Builder) saveFlagPresetsLocked() {
	os.MkdirAll(filepath.Dir(b.flagPresetsPath()), 0o755)
	data, err := json.MarshalIndent(b.flagPresets, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(b.flagPresetsPath(), data, 0o644)
}

// FlagPresets returns the saved flag sets for a profile ("" = all),
// sorted by name.
func (b *Builder) FlagPresets(profile string) []FlagPreset {
	b.fpMu.Lock()
	defer b.fpMu.Unlock()
	b.loadFlagPresetsLocked()
	var out []FlagPreset
	for _, p := range b.flagPresets {
		if profile == "" || p.Profile == profile {
			out = append(out, p)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// FindFlagPreset returns the saved flag set with the given name.
func (b *Builder) FindFlagPreset(name string) (FlagPreset, bool) {
	b.fpMu.Lock()
	defer b.fpMu.Unlock()
	b.loadFlagPresetsLocked()
	for _, p := range b.flagPresets {
		if p.Name == name {
			return p, true
		}
	}
	return FlagPreset{}, false
}

// SaveFlagPreset upserts a flag set — saving under an existing name
// updates it, which is the natural "update my favorite" flow. The name
// follows the build-tag rules so it can label builds directly. A preset
// without a current Version (one restored from an older backup) is
// migrated first; the build form's save sets the current Version.
func (b *Builder) SaveFlagPreset(p FlagPreset) error {
	if !validTagRE.MatchString(p.Name) {
		return fmt.Errorf("invalid name %q: lowercase letters, digits, and hyphens", p.Name)
	}
	if p.Profile == "" {
		return errors.New("a flag preset needs a profile")
	}
	migrateFlagPreset(&p)
	b.fpMu.Lock()
	defer b.fpMu.Unlock()
	b.loadFlagPresetsLocked()
	for i := range b.flagPresets {
		if b.flagPresets[i].Name == p.Name {
			b.flagPresets[i] = p
			b.saveFlagPresetsLocked()
			return nil
		}
	}
	b.flagPresets = append(b.flagPresets, p)
	b.saveFlagPresetsLocked()
	return nil
}

// DeleteFlagPreset removes a saved flag set, reporting whether it existed.
func (b *Builder) DeleteFlagPreset(name string) bool {
	b.fpMu.Lock()
	defer b.fpMu.Unlock()
	b.loadFlagPresetsLocked()
	for i, p := range b.flagPresets {
		if p.Name == name {
			b.flagPresets = append(b.flagPresets[:i], b.flagPresets[i+1:]...)
			b.saveFlagPresetsLocked()
			return true
		}
	}
	return false
}
