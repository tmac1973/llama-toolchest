package enginenotes

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
	"github.com/tmac1973/llama-toolchest/internal/serverlog"
)

// KindRouterStart is a router start. The tracker cannot see one in the
// log, so the server records it where it starts the router.
const KindRouterStart OutcomeKind = "router_start"

// SourceLlamaServer marks events about llama-server. Generation backends
// will record under their own source later; the field keeps room for
// them in the same file.
const SourceLlamaServer = "llama-server"

// The bounds that keep the file small. Crash records are the ones worth
// keeping longest, so they have their own cap and are the last to go.
const (
	maxEvents      = 5000
	maxCrashEvents = 100
)

// fileVersion is the format of engine-notes.json.
const fileVersion = 1

// saveEvery is the longest a change waits before it is written. Usage
// changes on every request; writing each one would be a write per
// request for figures nobody reads that fast.
const saveEvery = 5 * time.Second

// hourKey is the format of a usage bucket's key: its UTC hour.
const hourKey = "2006-01-02T15"

// Event is one thing that happened, as it is saved.
type Event struct {
	Time   time.Time   `json:"time"`
	Source string      `json:"source"`
	Kind   OutcomeKind `json:"kind"`

	// ModelID is the registry id and ModelName the display name when the
	// event was recorded, so a model deleted since still reads well.
	// RouterName is the router's name= for it.
	ModelID    string `json:"model_id,omitempty"`
	ModelName  string `json:"model_name,omitempty"`
	RouterName string `json:"router_name,omitempty"`
	Port       string `json:"port,omitempty"`

	ExitStatus    int     `json:"exit_status,omitempty"`
	LoadSeconds   float64 `json:"load_seconds,omitempty"`
	LoadedSeconds float64 `json:"loaded_seconds,omitempty"`

	// Build is the llama.cpp build that was running, and GPUs the GPU
	// assignment the model was launched with, as the UI labels it.
	Build string `json:"build,omitempty"`
	GPUs  string `json:"gpus,omitempty"`

	// StartReason is "user" or "job", on a router start.
	StartReason string `json:"start_reason,omitempty"`
	// JobID and JobName name the benchmark or Autoconfigure job that
	// held the router, if one did.
	JobID   string `json:"job_id,omitempty"`
	JobName string `json:"job_name,omitempty"`

	RouterError string   `json:"router_error,omitempty"`
	ErrorLines  []string `json:"error_lines,omitempty"`
	Tail        []string `json:"tail,omitempty"`
}

// IsFailure reports whether the event is a crash or a failed load: the
// kinds the strip warns about and that keep their log lines.
func (e Event) IsFailure() bool {
	return e.Kind == KindCrash || e.Kind == KindLoadFailed
}

// Usage is one model's requests and tokens in one hour.
type Usage struct {
	Requests     int `json:"requests"`
	Failed       int `json:"failed"`
	PromptTokens int `json:"prompt_tokens"`
	GenTokens    int `json:"gen_tokens"`
}

// fileData is engine-notes.json.
type fileData struct {
	Version       int                          `json:"version"`
	CountingSince time.Time                    `json:"counting_since"`
	Events        []Event                      `json:"events"`
	Usage         map[string]map[string]*Usage `json:"usage"`
}

// Store keeps events and usage in memory and in engine-notes.json. Its
// methods are safe for concurrent use.
type Store struct {
	path  string
	now   func() time.Time
	write func(path string, data []byte) error

	mu    sync.Mutex
	data  fileData
	dirty bool
	// saveMu makes saves one at a time, so a copy taken earlier is never
	// written over one taken later.
	saveMu sync.Mutex

	stop chan struct{}
	done chan struct{}
	once sync.Once
}

// Open loads the store saved at path, or starts an empty one when there
// is none. A file that cannot be read is moved aside and the store starts
// empty: losing the history is better than not starting.
func Open(path string, now func() time.Time) (*Store, error) {
	return open(path, now, saveEvery, atomicfile.Write)
}

func open(path string, now func() time.Time, every time.Duration, write func(string, []byte) error) (*Store, error) {
	s := &Store{
		path:  path,
		now:   now,
		write: write,
		stop:  make(chan struct{}),
		done:  make(chan struct{}),
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	go s.saveLoop(every)
	return s, nil
}

func (s *Store) empty() fileData {
	return fileData{
		Version:       fileVersion,
		CountingSince: s.now().UTC(),
		Usage:         map[string]map[string]*Usage{},
	}
}

func (s *Store) load() error {
	raw, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		s.data = s.empty()
		return nil
	}
	if err != nil {
		return fmt.Errorf("read engine notes: %w", err)
	}
	var d fileData
	reason := ""
	if err := json.Unmarshal(raw, &d); err != nil {
		reason = err.Error()
	} else if d.Version != fileVersion {
		reason = fmt.Sprintf("unknown version %d", d.Version)
	}
	if reason != "" {
		bad := fmt.Sprintf("%s.bad-%d", s.path, s.now().Unix())
		if err := os.Rename(s.path, bad); err != nil {
			return fmt.Errorf("move aside unreadable engine notes: %w", err)
		}
		slog.Warn("engine notes file could not be read; starting a new one", "path", s.path, "moved_to", bad, "reason", reason)
		s.data = s.empty()
		return nil
	}
	if d.Usage == nil {
		d.Usage = map[string]map[string]*Usage{}
	}
	s.data = d
	return nil
}

// Record adds an event. Fields that only a failure keeps are dropped from
// other kinds, and long lists are cut to their bounds.
func (s *Store) Record(e Event) {
	if e.Time.IsZero() {
		e.Time = s.now()
	}
	e.Time = e.Time.UTC()
	if e.Source == "" {
		e.Source = SourceLlamaServer
	}
	if e.IsFailure() {
		if len(e.Tail) > tailLines {
			e.Tail = e.Tail[len(e.Tail)-tailLines:]
		}
		if len(e.ErrorLines) > serverlog.MaxErrorLines {
			e.ErrorLines = e.ErrorLines[:serverlog.MaxErrorLines]
		}
	} else {
		e.Tail, e.ErrorLines = nil, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Events = append(s.data.Events, e)
	s.trim()
	s.dirty = true
}

// trim drops the oldest events past the bounds: failures past their own
// cap first, then other events until the total fits. Failures are never
// dropped to make room for the others.
func (s *Store) trim() {
	failures := 0
	for _, e := range s.data.Events {
		if e.IsFailure() {
			failures++
		}
	}
	for failures > maxCrashEvents {
		s.dropOldest(Event.IsFailure)
		failures--
	}
	for len(s.data.Events) > maxEvents {
		if !s.dropOldest(func(e Event) bool { return !e.IsFailure() }) {
			return
		}
	}
}

func (s *Store) dropOldest(match func(Event) bool) bool {
	oldest := -1
	for i, e := range s.data.Events {
		if match(e) && (oldest < 0 || e.Time.Before(s.data.Events[oldest].Time)) {
			oldest = i
		}
	}
	if oldest < 0 {
		return false
	}
	s.data.Events = append(s.data.Events[:oldest], s.data.Events[oldest+1:]...)
	return true
}

// AddUsage counts one request to a model in the current hour.
func (s *Store) AddUsage(modelID string, promptTokens, genTokens int, failed bool) {
	if modelID == "" {
		return
	}
	key := s.now().UTC().Format(hourKey)

	s.mu.Lock()
	defer s.mu.Unlock()
	hours := s.data.Usage[modelID]
	if hours == nil {
		hours = map[string]*Usage{}
		s.data.Usage[modelID] = hours
	}
	u := hours[key]
	if u == nil {
		u = &Usage{}
		hours[key] = u
	}
	u.Requests++
	if failed {
		u.Failed++
	}
	u.PromptTokens += promptTokens
	u.GenTokens += genTokens
	s.dirty = true
}

// DeleteOlderThan removes events before cutoff and usage hours that start
// before it, saves at once, and says how much went. Counting then starts
// at cutoff, since nothing older is left.
func (s *Store) DeleteOlderThan(cutoff time.Time) (events, hours int, err error) {
	cutoff = cutoff.UTC()
	s.mu.Lock()
	kept := s.data.Events[:0:0]
	for _, e := range s.data.Events {
		if e.Time.Before(cutoff) {
			events++
			continue
		}
		kept = append(kept, e)
	}
	s.data.Events = kept
	for id, byHour := range s.data.Usage {
		for key := range byHour {
			if t, err := time.Parse(hourKey, key); err != nil || t.Before(cutoff) {
				delete(byHour, key)
				hours++
			}
		}
		if len(byHour) == 0 {
			delete(s.data.Usage, id)
		}
	}
	if s.data.CountingSince.Before(cutoff) {
		s.data.CountingSince = cutoff
	}
	s.dirty = true
	s.mu.Unlock()
	return events, hours, s.Save()
}

// DeleteAll removes everything, starts counting again from now, and
// saves at once.
func (s *Store) DeleteAll() (events, hours int, err error) {
	s.mu.Lock()
	events = len(s.data.Events)
	for _, byHour := range s.data.Usage {
		hours += len(byHour)
	}
	s.data = s.empty()
	s.dirty = true
	s.mu.Unlock()
	return events, hours, s.Save()
}

// Save writes the store if anything changed since the last write.
func (s *Store) Save() error {
	s.saveMu.Lock()
	defer s.saveMu.Unlock()
	s.mu.Lock()
	if !s.dirty {
		s.mu.Unlock()
		return nil
	}
	raw, err := json.Marshal(s.data)
	s.dirty = false
	s.mu.Unlock()
	if err != nil {
		return fmt.Errorf("encode engine notes: %w", err)
	}
	if err := s.write(s.path, raw); err != nil {
		s.mu.Lock()
		s.dirty = true // try again on the next tick
		s.mu.Unlock()
		return fmt.Errorf("save engine notes: %w", err)
	}
	return nil
}

func (s *Store) saveLoop(every time.Duration) {
	defer close(s.done)
	tick := time.NewTicker(every)
	defer tick.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-tick.C:
			if err := s.Save(); err != nil {
				slog.Warn("engine notes not saved", "error", err)
			}
		}
	}
}

// Close stops the background saving and writes what is left. Calling it
// again does nothing.
func (s *Store) Close() error {
	var err error
	s.once.Do(func() {
		close(s.stop)
		<-s.done
		err = s.Save()
	})
	return err
}
