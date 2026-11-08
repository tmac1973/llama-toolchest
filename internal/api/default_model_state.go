package api

import (
	"encoding/json"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
)

// defaultModelState remembers the chat model used last, so the virtual
// model name (config.DefaultModelConfig) has a target after a restart
// with nothing loaded. It is kept in <DataDir>/default-model.json.
type defaultModelState struct {
	path string

	mu   sync.Mutex
	last lastChatModel
	// writes counts the file writes, so a test can tell a note that
	// changed nothing from one that wrote.
	writes int
}

// lastChatModel is the chat model used last.
type lastChatModel struct {
	ID     string    `json:"id"` // registry ID, or fork instance ID when Fork
	Fork   bool      `json:"fork,omitempty"`
	UsedAt time.Time `json:"used_at"`
}

// loadDefaultModelState reads the state saved at path. A missing or
// unreadable file gives an empty state: the next chat request fills it.
func loadDefaultModelState(path string) *defaultModelState {
	s := &defaultModelState{path: path}
	data, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) {
			slog.Warn("default model state unreadable, starting empty", "path", path, "error", err)
		}
		return s
	}
	if err := json.Unmarshal(data, &s.last); err != nil {
		slog.Warn("default model state unreadable, starting empty", "path", path, "error", err)
		s.last = lastChatModel{}
	}
	return s
}

// get returns the chat model used last; its ID is "" when there is none.
func (s *defaultModelState) get() lastChatModel {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

// note records model id as used at at. The file is written only when the
// model changes, so a run of requests to one model writes nothing; the
// time on disk is then the first use of the run. A failed write is
// logged, and the new value is still kept in memory.
func (s *defaultModelState) note(id string, fork bool, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	changed := s.last.ID != id || s.last.Fork != fork
	s.last = lastChatModel{ID: id, Fork: fork, UsedAt: at}
	if !changed {
		return
	}
	data, err := json.MarshalIndent(s.last, "", "  ")
	if err == nil {
		s.writes++
		err = atomicfile.Write(s.path, data)
	}
	if err != nil {
		slog.Warn("default model state not saved", "path", s.path, "error", err)
	}
}
