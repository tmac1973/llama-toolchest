package generation

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// SchemaVersion is the generation.json format this build reads and writes.
const SchemaVersion = 1

// Store errors that callers check for.
var (
	ErrNotFound = errors.New("not found")
	ErrIDTaken  = errors.New("ID already in use")
	ErrInUse    = errors.New("in use by an instance")
	ErrNoPort   = errors.New("no free port in the generation port range")
	ErrPortUsed = errors.New("already used by another instance")
)

type storeFile struct {
	SchemaVersion int        `json:"schema_version"`
	ModelSets     []ModelSet `json:"model_sets"`
	Instances     []Instance `json:"instances"`
}

// Store persists model sets and backend instances to
// <data>/config/generation.json. Nothing is written until the first change,
// so installs that never use generation backends gain no file.
type Store struct {
	path string

	mu   sync.RWMutex
	data storeFile
}

// StorePath returns where the store lives under a data directory.
func StorePath(dataDir string) string {
	return filepath.Join(dataDir, "config", "generation.json")
}

// Load reads the store under dataDir. A missing file is an empty store.
func Load(dataDir string) (*Store, error) {
	s := &Store{path: StorePath(dataDir), data: storeFile{SchemaVersion: SchemaVersion}}
	raw, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", s.path, err)
	}
	var f storeFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("reading %s: %w", s.path, err)
	}
	if f.SchemaVersion != SchemaVersion {
		return nil, fmt.Errorf("%s has schema version %d; this build reads version %d",
			s.path, f.SchemaVersion, SchemaVersion)
	}
	s.data = f
	return s, nil
}

// saveLocked writes the store. Callers must hold mu for writing.
func (s *Store) saveLocked() error {
	raw, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return fmt.Errorf("saving generation store: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("saving generation store: %w", err)
	}
	if err := atomicfile.WriteMode(s.path, raw, 0o600); err != nil {
		return fmt.Errorf("saving generation store to %s: %w", s.path, err)
	}
	return nil
}

// Sets and instances go in and out of the store as deep copies: their
// slices must not share memory with the store's own, or a caller changing
// a set (a download recording a file's path) writes into data another
// goroutine is reading (a page being rendered).

func (m ModelSet) clone() ModelSet {
	m.Components = slices.Clone(m.Components)
	m.LaunchArgs = slices.Clone(m.LaunchArgs)
	m.Capabilities = slices.Clone(m.Capabilities)
	return m
}

func (inst Instance) clone() Instance {
	inst.ExtraArgs = slices.Clone(inst.ExtraArgs)
	inst.MeasuredArgs = slices.Clone(inst.MeasuredArgs)
	inst.MeasurePoints = slices.Clone(inst.MeasurePoints)
	return inst
}

// ModelSets returns a copy of all model sets.
func (s *Store) ModelSets() []ModelSet {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]ModelSet, len(s.data.ModelSets))
	for i, m := range s.data.ModelSets {
		out[i] = m.clone()
	}
	return out
}

// ModelSet returns the set with the given ID.
func (s *Store) ModelSet(id string) (ModelSet, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, m := range s.data.ModelSets {
		if m.ID == id {
			return m.clone(), nil
		}
	}
	return ModelSet{}, fmt.Errorf("model set %q: %w", id, ErrNotFound)
}

// AddModelSet stores a new set. Its ID must not be taken.
func (s *Store) AddModelSet(m ModelSet) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.setIndexLocked(m.ID) >= 0 {
		return fmt.Errorf("model set %q: %w", m.ID, ErrIDTaken)
	}
	s.data.ModelSets = append(s.data.ModelSets, m.clone())
	return s.saveLocked()
}

// UpdateModelSet replaces the stored set with the same ID.
func (s *Store) UpdateModelSet(m ModelSet) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.setIndexLocked(m.ID)
	if i < 0 {
		return fmt.Errorf("model set %q: %w", m.ID, ErrNotFound)
	}
	s.data.ModelSets[i] = m.clone()
	return s.saveLocked()
}

// DeleteModelSet removes a set. It is refused while an instance uses it.
func (s *Store) DeleteModelSet(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.setIndexLocked(id)
	if i < 0 {
		return fmt.Errorf("model set %q: %w", id, ErrNotFound)
	}
	for _, inst := range s.data.Instances {
		if inst.ModelSetID == id {
			return fmt.Errorf("model set %q is used by instance %q: %w", id, inst.ID, ErrInUse)
		}
	}
	s.data.ModelSets = slices.Delete(s.data.ModelSets, i, i+1)
	return s.saveLocked()
}

// Instances returns a copy of all instances.
func (s *Store) Instances() []Instance {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Instance, len(s.data.Instances))
	for i, inst := range s.data.Instances {
		out[i] = inst.clone()
	}
	return out
}

// Instance returns the instance with the given ID.
func (s *Store) Instance(id string) (Instance, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, inst := range s.data.Instances {
		if inst.ID == id {
			return inst.clone(), nil
		}
	}
	return Instance{}, fmt.Errorf("instance %q: %w", id, ErrNotFound)
}

// AddInstance stores a new instance. Its ID must not be taken and its
// model set must exist.
func (s *Store) AddInstance(inst Instance) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.instIndexLocked(inst.ID) >= 0 {
		return fmt.Errorf("instance %q: %w", inst.ID, ErrIDTaken)
	}
	if s.setIndexLocked(inst.ModelSetID) < 0 {
		return fmt.Errorf("model set %q: %w", inst.ModelSetID, ErrNotFound)
	}
	s.data.Instances = append(s.data.Instances, inst.clone())
	return s.saveLocked()
}

// UpdateInstance replaces the stored instance with the same ID.
func (s *Store) UpdateInstance(inst Instance) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.instIndexLocked(inst.ID)
	if i < 0 {
		return fmt.Errorf("instance %q: %w", inst.ID, ErrNotFound)
	}
	s.data.Instances[i] = inst.clone()
	return s.saveLocked()
}

// EditInstance changes the stored instance with this ID through fn, in one
// step, so that a change made meanwhile elsewhere is not overwritten.
func (s *Store) EditInstance(id string, fn func(*Instance)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.instIndexLocked(id)
	if i < 0 {
		return fmt.Errorf("instance %q: %w", id, ErrNotFound)
	}
	inst := s.data.Instances[i].clone()
	fn(&inst)
	s.data.Instances[i] = inst
	return s.saveLocked()
}

// DeleteInstance removes an instance.
func (s *Store) DeleteInstance(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.instIndexLocked(id)
	if i < 0 {
		return fmt.Errorf("instance %q: %w", id, ErrNotFound)
	}
	s.data.Instances = slices.Delete(s.data.Instances, i, i+1)
	return s.saveLocked()
}

// AssignPort returns the lowest port in [start, start+count) that no
// instance uses.
func (s *Store) AssignPort(start, count int) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	used := make(map[int]bool, len(s.data.Instances))
	for _, inst := range s.data.Instances {
		used[inst.Port] = true
	}
	for p := start; p < start+count; p++ {
		if !used[p] {
			return p, nil
		}
	}
	return 0, fmt.Errorf("%d–%d: %w", start, start+count-1, ErrNoPort)
}

// CheckPort reports whether the instance id can use port: the port must be
// in [start, start+count) and no other instance may use it. id is empty
// for an instance that is not stored yet.
func (s *Store) CheckPort(id string, port, start, count int) error {
	if port < start || port >= start+count {
		return fmt.Errorf("port %d is outside the generation port range %d–%d", port, start, start+count-1)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, inst := range s.data.Instances {
		if inst.Port == port && inst.ID != id {
			return fmt.Errorf("port %d is %w (%s)", port, ErrPortUsed, inst.ID)
		}
	}
	return nil
}

func (s *Store) setIndexLocked(id string) int {
	return slices.IndexFunc(s.data.ModelSets, func(m ModelSet) bool { return m.ID == id })
}

func (s *Store) instIndexLocked(id string) int {
	return slices.IndexFunc(s.data.Instances, func(i Instance) bool { return i.ID == id })
}

// ModelsDir returns where generation model files are downloaded: a
// gen-models folder inside the LLM models directory, so they sit with the
// other model files (in a container, on the host's models mount). The LLM
// registry skips that folder, so it never takes them for language models.
// It is not created here; the downloader creates it when it is first
// needed.
func ModelsDir(cfg *config.Config) string {
	return filepath.Join(cfg.ModelsPath(), models.GenModelsDirName)
}
