package autotune

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
)

// Status values of an autotune run.
const (
	StatusPlanned     = "planned"
	StatusRunning     = "running"
	StatusInterrupted = "interrupted" // the server stopped while it ran
	StatusCancelled   = "cancelled"
	StatusDone        = "done"
	StatusFailed      = "failed"
)

// Stage keys, in the order they run.
const (
	StageBatch      = "batch"
	StageSpec       = "spec"
	StageSpecParams = "spec-params"
	StageConfirm    = "confirm"
)

// StageOrder is the order the stages run in, with the title each one
// shows.
var StageOrder = []struct{ Key, Title string }{
	{StageBatch, "Batch sizes and attention"},
	{StageSpec, "Speculative decoding"},
	{StageSpecParams, "Fine-tuning the draft settings"},
	{StageConfirm, "Confirming the finalists"},
}

// StageTitle is a stage's title, or the key when it is unknown.
func StageTitle(key string) string {
	for _, s := range StageOrder {
		if s.Key == key {
			return s.Title
		}
	}
	return key
}

// Autotune is one run: what it measures, how far it has got, and what it
// found.
type Autotune struct {
	ID      string `json:"id"`
	ModelID string `json:"model_id"`
	// BaseProfile is the saved profile the run measures from. Its
	// context size, KV cache type and sampling are kept throughout: they
	// change what the model answers, and autotune only changes speed.
	BaseProfile string  `json:"base_profile"`
	UseCase     UseCase `json:"use_case"`
	BuildID     string  `json:"build_id,omitempty"`
	Status      string  `json:"status"`

	Stages []StageRecord `json:"stages,omitempty"`
	// Skipped lists what could not be measured and why, in plain
	// language: "EAGLE3: no EAGLE3 head is installed for this model".
	Skipped []string `json:"skipped,omitempty"`
	// Results is what the run concluded, by goal.
	Results map[Goal]Outcome `json:"results,omitempty"`
	// SecondsPerCell is measured after the first stage and used for the
	// time remaining.
	SecondsPerCell float64 `json:"seconds_per_cell,omitempty"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	Error     string    `json:"error,omitempty"`
}

// StageRecord is one stage of a run.
type StageRecord struct {
	Key    string `json:"key"`
	Title  string `json:"title"`
	JobID  string `json:"job_id,omitempty"`
	Status string `json:"status"`
	// Failed lists the candidates whose runs failed, so a setting that
	// cannot run on this machine is reported rather than silently absent.
	Failed []FailedCell `json:"failed,omitempty"`
	// Finalists are carried into the next stage: the winner for each
	// goal, de-duplicated.
	Finalists []Candidate `json:"finalists,omitempty"`
	// Candidates is everything this stage measured, for the screens.
	Candidates []Candidate `json:"candidates,omitempty"`
}

// FailedCell is one candidate that could not be measured.
type FailedCell struct {
	Label string `json:"label"`
	Error string `json:"error"`
}

// Outcome is what a run concluded for one goal.
type Outcome struct {
	Goal Goal `json:"goal"`
	// Saved is false when the starting profile was already the fastest,
	// in which case no profile is written.
	Saved       bool      `json:"saved"`
	ProfileName string    `json:"profile_name,omitempty"`
	Winner      Candidate `json:"winner"`
	Baseline    Candidate `json:"baseline"`
	// Message is the one-line summary the results screen shows.
	Message string `json:"message"`
}

// Stage returns the record of a stage, and whether it exists.
func (a *Autotune) Stage(key string) (*StageRecord, bool) {
	for i := range a.Stages {
		if a.Stages[i].Key == key {
			return &a.Stages[i], true
		}
	}
	return nil, false
}

// SetStage inserts or replaces a stage record, keeping stage order.
func (a *Autotune) SetStage(rec StageRecord) {
	if rec.Title == "" {
		rec.Title = StageTitle(rec.Key)
	}
	for i := range a.Stages {
		if a.Stages[i].Key == rec.Key {
			a.Stages[i] = rec
			return
		}
	}
	a.Stages = append(a.Stages, rec)
	order := map[string]int{}
	for i, s := range StageOrder {
		order[s.Key] = i
	}
	sort.SliceStable(a.Stages, func(i, j int) bool { return order[a.Stages[i].Key] < order[a.Stages[j].Key] })
}

// Finalists returns the candidates a stage carried forward, or the
// baseline-only list before that stage has run.
func (a *Autotune) Finalists(key string) []Candidate {
	if st, ok := a.Stage(key); ok && len(st.Finalists) > 0 {
		return st.Finalists
	}
	return nil
}

// autotuneSchemaVersion is the layout this build writes. Read on load,
// like the model registry and the benchmark store: a file from a newer
// build is served but never overwritten.
const autotuneSchemaVersion = 1

type autotuneFile struct {
	SchemaVersion int         `json:"schema_version"`
	Runs          []*Autotune `json:"runs"`
}

// ErrStoreReadOnly is matched by errors.Is on every refusal from a
// read-only store.
var ErrStoreReadOnly = errors.New("autotune history is read-only")

type readOnlyError struct{ reason string }

func (e readOnlyError) Error() string        { return e.reason }
func (e readOnlyError) Is(target error) bool { return target == ErrStoreReadOnly }

// Store keeps autotune runs in <dataDir>/config/autotune.json.
type Store struct {
	mu       sync.RWMutex
	dataDir  string
	runs     []*Autotune
	readOnly string
}

// NewStore loads the stored runs. A run left "running" by a server that
// stopped is marked interrupted, so it can be resumed rather than looking
// like it is still going.
func NewStore(dataDir string) *Store {
	s := &Store{dataDir: dataDir}
	s.load()
	return s
}

func (s *Store) path() string { return filepath.Join(s.dataDir, "config", "autotune.json") }

func (s *Store) load() {
	data, err := os.ReadFile(s.path())
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			s.readOnly = fmt.Sprintf("autotune.json could not be read (%v). "+
				"Autotune results will not be saved until the file can be read, "+
				"so that nothing overwrites it.", err)
			slog.Error("failed to read autotune history; store is read-only", "error", err)
		}
		return
	}
	var file autotuneFile
	if err := json.Unmarshal(data, &file); err != nil {
		s.readOnly = fmt.Sprintf("autotune.json could not be parsed (%v). "+
			"Autotune results will not be saved until the file is fixed, "+
			"so that nothing overwrites it.", err)
		slog.Error("failed to parse autotune history; store is read-only", "error", err)
		return
	}
	s.runs = file.Runs
	if file.SchemaVersion > autotuneSchemaVersion {
		s.readOnly = fmt.Sprintf("autotune.json was written by a newer version of "+
			"llama-toolchest (schema %d; this version reads up to %d). "+
			"Autotune results will not be saved until llama-toolchest is upgraded.",
			file.SchemaVersion, autotuneSchemaVersion)
		slog.Error("autotune history is from a newer build; store is read-only",
			"schema_version", file.SchemaVersion, "supported", autotuneSchemaVersion)
		return
	}
	dirty := false
	for _, r := range s.runs {
		if r.Status == StatusRunning {
			// The process that was running it is gone.
			r.Status = StatusInterrupted
			dirty = true
		}
	}
	if dirty {
		s.save()
	}
}

// ReadOnlyReason says why the store refuses to save, or "" when it is
// writable.
func (s *Store) ReadOnlyReason() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.readOnly
}

func (s *Store) writableLocked() error {
	if s.readOnly != "" {
		return readOnlyError{s.readOnly}
	}
	return nil
}

// save writes the file. Callers hold s.mu.
func (s *Store) save() error {
	if err := s.writableLocked(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(autotuneFile{SchemaVersion: autotuneSchemaVersion, Runs: s.runs}, "", "  ")
	if err != nil {
		return fmt.Errorf("saving autotune.json: %w", err)
	}
	if err := atomicfile.Write(s.path(), data); err != nil {
		return fmt.Errorf("saving autotune.json: %w", err)
	}
	return nil
}

// maxStoredRuns bounds the history. A run holds every candidate it
// measured, so the file would otherwise grow without limit.
const maxStoredRuns = 50

// Save stores a run, replacing one with the same ID.
func (s *Store) Save(rec *Autotune) error {
	if rec == nil || rec.ID == "" {
		return errors.New("an autotune run needs an ID")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.writableLocked(); err != nil {
		return err
	}
	rec.UpdatedAt = time.Now().UTC()
	for i, r := range s.runs {
		if r.ID == rec.ID {
			s.runs[i] = rec
			return s.save()
		}
	}
	s.runs = append(s.runs, rec)
	sort.SliceStable(s.runs, func(i, j int) bool { return s.runs[i].CreatedAt.After(s.runs[j].CreatedAt) })
	if len(s.runs) > maxStoredRuns {
		s.runs = s.runs[:maxStoredRuns]
	}
	return s.save()
}

// Get returns a run by ID.
func (s *Store) Get(id string) (*Autotune, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, r := range s.runs {
		if r.ID == id {
			return r, true
		}
	}
	return nil, false
}

// List returns every stored run, newest first.
func (s *Store) List() []*Autotune {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]*Autotune, len(s.runs))
	copy(out, s.runs)
	return out
}

// LatestForModel returns the most recent run for a model.
func (s *Store) LatestForModel(modelID string) (*Autotune, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var best *Autotune
	for _, r := range s.runs {
		if r.ModelID != modelID {
			continue
		}
		if best == nil || r.CreatedAt.After(best.CreatedAt) {
			best = r
		}
	}
	return best, best != nil
}

// Active returns the run that is going, if any.
func (s *Store) Active() (*Autotune, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, r := range s.runs {
		if r.Status == StatusRunning || r.Status == StatusPlanned {
			return r, true
		}
	}
	return nil, false
}
