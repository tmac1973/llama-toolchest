package models

import (
	"errors"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
	"time"
)

// ConfigProfile is a named snapshot of one model's launch config, saved on
// demand and restored on demand. It is not a backup and not an undo step:
// the live config keeps autosaving as it always has, and a profile changes
// only when someone saves one.
//
// The scope is one model, deliberately: "long context" means a different
// context size on a 27B than on a 4B, and a global profile would have to
// either carry fields most models should not take or become a partial
// overlay with merge rules.
//
// A profile belongs to a model identity — the repository and the file
// name — not to a registry ID. The registry ID of the same file differs
// between a download and a disk scan, and a model deleted to free space
// and pulled again is exactly when a saved profile is most useful. For
// the same reason profiles live on the registry envelope, not on Model,
// and are not removed when their model is.
type ConfigProfile struct {
	RepoID   string `json:"repo_id"`  // Model.ModelID, e.g. "unsloth/Qwen3.5-9B-GGUF"
	Filename string `json:"filename"` // Model.Filename (the first shard of a split model)
	Name     string `json:"name"`

	// Config holds every launch and sampling setting. Enabled, Aliases and
	// ActiveProfile are always stored empty: they name the model and switch
	// it on or off rather than tune it, so restoring a profile never
	// renames or disables a model.
	Config ModelConfig `json:"config"`

	SavedAt time.Time `json:"saved_at"`
	// BuildID is the llama.cpp build that was active when the profile was
	// saved. Speculative decoding methods and flags depend on the build, so
	// restoring on another one is worth a warning.
	BuildID string `json:"build_id,omitempty"`
	// Source is who wrote the profile: ProfileSourceUser, …Autoconfig or
	// …Autotune.
	Source string `json:"source"`
	// Notes explain individual settings, for profiles written by
	// autoconfigure or autotune.
	Notes []ProfileNote `json:"notes,omitempty"`
	// Measured is what autotune measured for this profile.
	Measured *ProfileMeasurement `json:"measured,omitempty"`
}

// Profile sources.
const (
	ProfileSourceUser       = "user"
	ProfileSourceAutoconfig = "autoconfig"
	ProfileSourceAutotune   = "autotune"
)

// ProfileNote is a plain-language reason for one setting.
type ProfileNote struct {
	Field  string `json:"field"` // ModelConfig JSON key, or "" for a general note
	Reason string `json:"reason"`
	// Origin is where the value came from: "hardware fit", "model card",
	// "publisher preset", "model file", "default" or "autotune".
	Origin string `json:"origin"`
}

// ProfileMeasurement is autotune's record of how a profile performed
// against the starting profile it was tuned from.
type ProfileMeasurement struct {
	Workload            string   `json:"workload"` // autotune use case: "chat", "code" or "mixed"
	PPTokPerSec         float64  `json:"pp_tok_per_sec"`
	TGTokPerSec         float64  `json:"tg_tok_per_sec"`
	ResponseSec         float64  `json:"response_sec"`
	BaselinePP          float64  `json:"baseline_pp"`
	BaselineTG          float64  `json:"baseline_tg"`
	BaselineResponseSec float64  `json:"baseline_response_sec"`
	Goals               []string `json:"goals"`
	AutotuneID          string   `json:"autotune_id"`
}

// MaxProfileNameLen bounds a name: long enough for "MTP + ngram, 128K
// context", short enough that the picker does not reflow the panel.
const MaxProfileNameLen = 64

var (
	ErrProfileName     = errors.New("a profile needs a name")
	ErrProfileNotFound = errors.New("no such profile")
)

// NormalizeProfileName is the stored display form of a name: trimmed,
// with runs of whitespace collapsed to one space.
func NormalizeProfileName(name string) string {
	return strings.Join(strings.Fields(name), " ")
}

// profileKey folds a name for identity only, so saving "Long Ctx" over
// "long  ctx" replaces that profile instead of adding a second entry the
// picker could not tell apart.
func profileKey(name string) string {
	return strings.ToLower(NormalizeProfileName(name))
}

func validateProfileName(name string) (string, error) {
	name = NormalizeProfileName(name)
	if name == "" {
		return "", ErrProfileName
	}
	if n := len([]rune(name)); n > MaxProfileNameLen {
		return "", fmt.Errorf("a profile name can be at most %d characters; this one is %d", MaxProfileNameLen, n)
	}
	return name, nil
}

// cloneConfig returns a copy of c that shares no memory with it, so a
// stored profile cannot change when the live config is edited in place.
func cloneConfig(c ModelConfig) ModelConfig {
	out := c
	if c.Aliases != nil {
		out.Aliases = append([]string(nil), c.Aliases...)
	}
	if c.ReasoningOverride != nil {
		r := *c.ReasoningOverride
		out.ReasoningOverride = &r
	}
	cloneF := func(p *float64) *float64 {
		if p == nil {
			return nil
		}
		v := *p
		return &v
	}
	out.Temperature = cloneF(c.Temperature)
	out.TopP = cloneF(c.TopP)
	out.MinP = cloneF(c.MinP)
	out.PresencePenalty = cloneF(c.PresencePenalty)
	out.RepeatPenalty = cloneF(c.RepeatPenalty)
	if c.TopK != nil {
		v := *c.TopK
		out.TopK = &v
	}
	return out
}

// profileFields is the part of a config a profile holds: a deep copy with
// the identity fields (Enabled, Aliases) and the ActiveProfile label
// cleared.
func profileFields(c ModelConfig) ModelConfig {
	out := cloneConfig(c)
	out.Enabled = false
	out.Aliases = nil
	out.ActiveProfile = ""
	return out
}

// ProfileEqual reports whether two configs hold the same profile settings,
// ignoring Enabled, Aliases and the ActiveProfile label. ModelConfig has
// pointer and slice fields, so == would compare addresses; DeepEqual
// compares what they point to.
func ProfileEqual(a, b ModelConfig) bool {
	return reflect.DeepEqual(profileFields(a), profileFields(b))
}

// profileIdentity is the (repository, file) pair a model's profiles are
// filed under.
func profileIdentity(m *Model) (repo, file string) {
	return m.ModelID, m.Filename
}

func (r *Registry) findProfileLocked(repo, file, name string) int {
	key := profileKey(name)
	for i, p := range r.data.Profiles {
		if p.RepoID == repo && p.Filename == file && profileKey(p.Name) == key {
			return i
		}
	}
	return -1
}

// modelAndConfigLocked resolves a registry ID to its model and live config.
func (r *Registry) modelAndConfigLocked(id string) (*Model, *ModelConfig, error) {
	m, ok := r.data.Models[id]
	if !ok {
		return nil, nil, fmt.Errorf("model not found: %s", id)
	}
	cfg, ok := r.data.Configs[id]
	if !ok {
		return nil, nil, fmt.Errorf("config not found: %s", id)
	}
	return m, cfg, nil
}

// upsertProfileLocked stores p, replacing a profile with the same identity
// and folded name, and keeps the list sorted for stable output.
func (r *Registry) upsertProfileLocked(p ConfigProfile) (replaced bool) {
	if i := r.findProfileLocked(p.RepoID, p.Filename, p.Name); i >= 0 {
		r.data.Profiles[i] = p
		return true
	}
	r.data.Profiles = append(r.data.Profiles, p)
	sort.Slice(r.data.Profiles, func(i, j int) bool {
		a, b := r.data.Profiles[i], r.data.Profiles[j]
		if a.RepoID != b.RepoID {
			return a.RepoID < b.RepoID
		}
		if a.Filename != b.Filename {
			return a.Filename < b.Filename
		}
		return profileKey(a.Name) < profileKey(b.Name)
	})
	return false
}

// SaveProfile stores a model's live config as a profile under name and
// makes it the model's active profile. An existing profile with the same
// name is replaced and reported through replaced: the caller says so
// afterwards rather than asking first.
func (r *Registry) SaveProfile(id, name, source, buildID string) (replaced bool, err error) {
	name, err = validateProfileName(name)
	if err != nil {
		return false, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	m, cfg, err := r.modelAndConfigLocked(id)
	if err != nil {
		return false, err
	}
	repo, file := profileIdentity(m)
	replaced = r.upsertProfileLocked(ConfigProfile{
		RepoID:   repo,
		Filename: file,
		Name:     name,
		Config:   profileFields(*cfg),
		SavedAt:  time.Now().UTC(),
		BuildID:  buildID,
		Source:   source,
	})
	cfg.ActiveProfile = name
	return replaced, r.save()
}

// SaveProfileFrom stores a profile built elsewhere (by autoconfigure or
// autotune) for the model with registry ID id. The live config is not
// touched. Name, identity and SavedAt are filled in here.
func (r *Registry) SaveProfileFrom(id, name string, p ConfigProfile) (replaced bool, err error) {
	name, err = validateProfileName(name)
	if err != nil {
		return false, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	m, ok := r.data.Models[id]
	if !ok {
		return false, fmt.Errorf("model not found: %s", id)
	}
	p.RepoID, p.Filename = profileIdentity(m)
	p.Name = name
	p.Config = profileFields(p.Config)
	if p.SavedAt.IsZero() {
		p.SavedAt = time.Now().UTC()
	}
	replaced = r.upsertProfileLocked(p)
	return replaced, r.save()
}

// Profiles returns copies of the profiles of the model with registry ID
// id, sorted by name.
func (r *Registry) Profiles(id string) []ConfigProfile {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.data.Models[id]
	if !ok {
		return nil
	}
	repo, file := profileIdentity(m)
	var out []ConfigProfile
	for _, p := range r.data.Profiles {
		if p.RepoID == repo && p.Filename == file {
			p.Config = cloneConfig(p.Config)
			out = append(out, p)
		}
	}
	return out
}

// GetProfile returns a copy of one profile of the model with registry ID
// id.
func (r *Registry) GetProfile(id, name string) (ConfigProfile, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, ok := r.data.Models[id]
	if !ok {
		return ConfigProfile{}, fmt.Errorf("model not found: %s", id)
	}
	repo, file := profileIdentity(m)
	i := r.findProfileLocked(repo, file, name)
	if i < 0 {
		return ConfigProfile{}, ErrProfileNotFound
	}
	p := r.data.Profiles[i]
	p.Config = cloneConfig(p.Config)
	return p, nil
}

// ApplyProfile copies a profile's settings over the live config of the
// model with registry ID id and makes it the active profile. The live
// config's Enabled and Aliases are kept.
func (r *Registry) ApplyProfile(id, name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return err
	}
	m, cfg, err := r.modelAndConfigLocked(id)
	if err != nil {
		return err
	}
	repo, file := profileIdentity(m)
	i := r.findProfileLocked(repo, file, name)
	if i < 0 {
		return ErrProfileNotFound
	}
	p := r.data.Profiles[i]
	next := cloneConfig(p.Config)
	next.Enabled = cfg.Enabled
	next.Aliases = cfg.Aliases
	next.ActiveProfile = p.Name
	NormalizeSpec(&next)
	*cfg = next
	return r.save()
}

// DeleteProfile removes one profile of the model with registry ID id. If
// it was the active profile, the label is cleared and the live config is
// left as it is.
func (r *Registry) DeleteProfile(id, name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return err
	}
	m, cfg, err := r.modelAndConfigLocked(id)
	if err != nil {
		return err
	}
	repo, file := profileIdentity(m)
	i := r.findProfileLocked(repo, file, name)
	if i < 0 {
		return ErrProfileNotFound
	}
	deleted := r.data.Profiles[i].Name
	r.data.Profiles = append(r.data.Profiles[:i], r.data.Profiles[i+1:]...)
	if profileKey(cfg.ActiveProfile) == profileKey(deleted) {
		cfg.ActiveProfile = ""
	}
	return r.save()
}

// ActiveProfileState reports the profile the live config was last saved
// as or restored from, and whether the live config has been edited since.
// name is "" when there is no active profile or it has been deleted.
func (r *Registry) ActiveProfileState(id string) (name string, edited bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	m, cfg, err := r.modelAndConfigLocked(id)
	if err != nil || cfg.ActiveProfile == "" {
		return "", false
	}
	repo, file := profileIdentity(m)
	i := r.findProfileLocked(repo, file, cfg.ActiveProfile)
	if i < 0 {
		return "", false
	}
	p := r.data.Profiles[i]
	return p.Name, !ProfileEqual(*cfg, p.Config)
}

// AllProfiles returns copies of every stored profile, including those
// whose model is not installed. Backup export uses it.
func (r *Registry) AllProfiles() []ConfigProfile {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]ConfigProfile, len(r.data.Profiles))
	for i, p := range r.data.Profiles {
		p.Config = cloneConfig(p.Config)
		out[i] = p
	}
	return out
}

// ImportProfile stores a profile restored from a backup, keyed by its own
// identity, whether or not that model is installed: a profile waits on the
// envelope until a matching file registers, with no claim step needed.
func (r *Registry) ImportProfile(p ConfigProfile) (replaced bool, err error) {
	name, err := validateProfileName(p.Name)
	if err != nil {
		return false, err
	}
	if p.RepoID == "" || p.Filename == "" {
		return false, errors.New("a profile needs a repository and a file name")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.writableLocked(); err != nil {
		return false, err
	}
	p.Name = name
	p.Config = profileFields(p.Config)
	replaced = r.upsertProfileLocked(p)
	return replaced, r.save()
}

// ValidateProfileConfig runs the checks a config must pass before it
// replaces a live config: the ones a config save runs, plus that every
// file it names still exists. A profile is a config from another time,
// and one pointing at a deleted draft model would otherwise fail minutes
// into a load, with nothing in the UI connecting the failure to the
// restore.
func ValidateProfileConfig(c ModelConfig) error {
	if err := c.ValidateBatchSizes(); err != nil {
		return err
	}
	if err := c.ValidateFlashAttention(); err != nil {
		return err
	}
	if err := c.ValidateSpec(); err != nil {
		return err
	}
	files := []struct {
		label, path string
		skip        bool
	}{
		{"vision projector (mmproj)", c.MmprojPath, c.MmprojDisabled},
		{"MTP draft head", c.MtpPath, c.MtpDisabled},
		{"draft model", c.DraftModelPath, false},
	}
	for _, f := range files {
		if f.path == "" || f.skip {
			continue
		}
		if st, err := os.Stat(f.path); err != nil || st.IsDir() {
			return fmt.Errorf("the %s this profile uses is no longer on disk: %s", f.label, f.path)
		}
	}
	return nil
}
