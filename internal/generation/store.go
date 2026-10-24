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
)

// SchemaVersion is the generation.json format this build reads and writes.
const SchemaVersion = 1

// Store errors that callers check for.
var (
	ErrNotFound = errors.New("not found")
	ErrIDTaken  = errors.New("ID already in use")
	ErrInUse    = errors.New("in use by an instance")
	ErrNoPort   = errors.New("no free port in the generation port range")
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

// ModelSets returns a copy of all model sets.
func (s *Store) ModelSets() []ModelSet {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.data.ModelSets)
}

// ModelSet returns the set with the given ID.
func (s *Store) ModelSet(id string) (ModelSet, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, m := range s.data.ModelSets {
		if m.ID == id {
			return m, nil
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
	s.data.ModelSets = append(s.data.ModelSets, m)
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
	s.data.ModelSets[i] = m
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
	return slices.Clone(s.data.Instances)
}

// Instance returns the instance with the given ID.
func (s *Store) Instance(id string) (Instance, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, inst := range s.data.Instances {
		if inst.ID == id {
			return inst, nil
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
	s.data.Instances = append(s.data.Instances, inst)
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

func (s *Store) setIndexLocked(id string) int {
	return slices.IndexFunc(s.data.ModelSets, func(m ModelSet) bool { return m.ID == id })
}

func (s *Store) instIndexLocked(id string) int {
	return slices.IndexFunc(s.data.Instances, func(i Instance) bool { return i.ID == id })
}

// ModelsDir returns where generation model files are downloaded: a
// gen-models directory next to the LLM models directory, so the LLM
// registry's scan never picks them up as language models. It is not
// created here; the downloader creates it when it is first needed.
func ModelsDir(cfg *config.Config) string {
	if cfg.ModelsDir != "" {
		return filepath.Join(filepath.Dir(filepath.Clean(cfg.ModelsDir)), "gen-models")
	}
	return filepath.Join(cfg.DataDir, "gen-models")
}
