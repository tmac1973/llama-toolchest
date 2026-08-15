package builder

import (
	"encoding/json"
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
// follows the build-tag rules so it can label builds directly.
func (b *Builder) SaveFlagPreset(p FlagPreset) error {
	if !validTagRE.MatchString(p.Name) {
		return fmt.Errorf("invalid name %q: lowercase letters, digits, and hyphens", p.Name)
	}
	if p.Profile == "" {
		return fmt.Errorf("a flag preset needs a profile")
	}
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
