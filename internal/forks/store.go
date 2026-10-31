package forks

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
)

// SchemaVersion is the forks.json format this build reads and writes.
const SchemaVersion = 1

// Store errors that callers check for.
var (
	ErrNotFound = errors.New("not found")
	ErrIDTaken  = errors.New("ID already in use")
	ErrBadID    = errors.New("may only have lowercase letters, digits, '.', '_' and '-'")
)

type storeFile struct {
	SchemaVersion int        `json:"schema_version"`
	ModelSets     []ModelSet `json:"model_sets"`
	Instances     []Instance `json:"instances"`
}

// Store persists fork model sets and fork instances to
// <data>/config/forks.json. Nothing is written until the first change,
// so installs that never use forks gain no file.
type Store struct {
	path string

	mu   sync.RWMutex
	data storeFile
}

// StorePath returns where the store lives under a data directory.
func StorePath(dataDir string) string {
	return filepath.Join(dataDir, "config", "forks.json")
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
		return fmt.Errorf("saving fork store: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("saving fork store: %w", err)
	}
	if err := atomicfile.WriteMode(s.path, raw, 0o600); err != nil {
		return fmt.Errorf("saving fork store to %s: %w", s.path, err)
	}
	return nil
}

// Instances go in and out of the store as deep copies, so a caller
// changing one never writes into memory another goroutine reads.
func (inst Instance) clone() Instance {
	inst.ExtraArgs = slices.Clone(inst.ExtraArgs)
	inst.ExtraEnv = slices.Clone(inst.ExtraEnv)
	return inst
}

// ModelSets returns a copy of all model sets.
func (s *Store) ModelSets() []ModelSet {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.data.ModelSets)
}

// ModelSet returns the model set of a fork.
func (s *Store) ModelSet(forkID string) (ModelSet, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if i := s.setIndexLocked(forkID); i >= 0 {
		return s.data.ModelSets[i], nil
	}
	return ModelSet{}, fmt.Errorf("model set for %q: %w", forkID, ErrNotFound)
}

// PutModelSet stores a fork's model set, replacing the one it had.
func (s *Store) PutModelSet(m ModelSet) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if i := s.setIndexLocked(m.ForkID); i >= 0 {
		s.data.ModelSets[i] = m
	} else {
		s.data.ModelSets = append(s.data.ModelSets, m)
	}
	return s.saveLocked()
}

// DeleteModelSet removes a fork's model set. The files are the caller's
// to delete.
func (s *Store) DeleteModelSet(forkID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.setIndexLocked(forkID)
	if i < 0 {
		return fmt.Errorf("model set for %q: %w", forkID, ErrNotFound)
	}
	s.data.ModelSets = slices.Delete(s.data.ModelSets, i, i+1)
	return s.saveLocked()
}

// Instances returns a copy of all fork instances.
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
	if i := s.instIndexLocked(id); i >= 0 {
		return s.data.Instances[i].clone(), nil
	}
	return Instance{}, fmt.Errorf("fork instance %q: %w", id, ErrNotFound)
}

// AddInstance stores a new instance. Its ID must be valid and not taken.
func (s *Store) AddInstance(inst Instance) error {
	if err := validateInstance(inst); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.instIndexLocked(inst.ID) >= 0 {
		return fmt.Errorf("fork instance %q: %w", inst.ID, ErrIDTaken)
	}
	s.data.Instances = append(s.data.Instances, inst.clone())
	return s.saveLocked()
}

// UpdateInstance replaces the stored instance with the same ID.
func (s *Store) UpdateInstance(inst Instance) error {
	if err := validateInstance(inst); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.instIndexLocked(inst.ID)
	if i < 0 {
		return fmt.Errorf("fork instance %q: %w", inst.ID, ErrNotFound)
	}
	s.data.Instances[i] = inst.clone()
	return s.saveLocked()
}

// DeleteInstance removes an instance.
func (s *Store) DeleteInstance(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	i := s.instIndexLocked(id)
	if i < 0 {
		return fmt.Errorf("fork instance %q: %w", id, ErrNotFound)
	}
	s.data.Instances = slices.Delete(s.data.Instances, i, i+1)
	return s.saveLocked()
}

func (s *Store) setIndexLocked(forkID string) int {
	return slices.IndexFunc(s.data.ModelSets, func(m ModelSet) bool { return m.ForkID == forkID })
}

func (s *Store) instIndexLocked(id string) int {
	return slices.IndexFunc(s.data.Instances, func(inst Instance) bool { return inst.ID == id })
}

var (
	instanceIDPattern = regexp.MustCompile(`^[a-z0-9._-]+$`)
	envNamePattern    = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
)

// deniedArgs are the options a fork instance gets from toolchest or its
// recipe's model files; a second copy in the extra arguments would win
// over them and break the instance.
var deniedArgs = []string{"--host", "--port", "-m", "--model", "-md", "--model-draft"}

// validateInstance checks what the store can check alone: the ID, the
// fork, the extra arguments and the environment lines. Ports and GPUs
// depend on the host and are checked by the caller.
func validateInstance(inst Instance) error {
	if !instanceIDPattern.MatchString(inst.ID) {
		return fmt.Errorf("instance name %q %w", inst.ID, ErrBadID)
	}
	if _, ok := Find(inst.ForkID); !ok {
		return fmt.Errorf("unknown fork %q", inst.ForkID)
	}
	if err := CheckExtraArgs(inst.ExtraArgs); err != nil {
		return err
	}
	return CheckEnv(inst.ExtraEnv)
}

// CheckExtraArgs refuses the options toolchest sets itself.
func CheckExtraArgs(args []string) error {
	for _, a := range args {
		name, _, _ := strings.Cut(a, "=")
		if slices.Contains(deniedArgs, name) {
			return fmt.Errorf("%s is set by llama-toolchest and cannot be given here", name)
		}
	}
	return nil
}

// CheckEnv checks NAME=value lines: NAME is upper case letters, digits
// and '_', and does not start with a digit.
func CheckEnv(lines []string) error {
	for _, line := range lines {
		name, _, ok := strings.Cut(line, "=")
		if !ok || !envNamePattern.MatchString(name) {
			return fmt.Errorf("%q is not NAME=value", line)
		}
	}
	return nil
}
