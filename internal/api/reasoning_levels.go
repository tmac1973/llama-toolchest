package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// reasoningLevelsTimeout caps one /props read. The model is loaded, so the
// answer is quick; a slow one is tried again on the next load.
const reasoningLevelsTimeout = 5 * time.Second

// reasoningLevelsReads is the set of models whose levels are being read,
// so a model that is seen loaded on every monitor tick is read once.
type reasoningLevelsReads struct {
	mu      sync.Mutex
	reading map[string]bool
}

// start marks id as being read, and reports false when it already is.
func (r *reasoningLevelsReads) start(id string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.reading == nil {
		r.reading = map[string]bool{}
	}
	if r.reading[id] {
		return false
	}
	r.reading[id] = true
	return true
}

func (r *reasoningLevelsReads) done(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.reading, id)
}

// noteModelLoaded reads, in the background, the reasoning effort levels of
// a model the router has loaded, unless they were already read on the
// active build. It costs nothing when there is nothing to read, so callers
// that see the model loaded can call it every time.
func (s *Server) noteModelLoaded(m *models.Model) {
	if m == nil || m.IsEmbedding() || !m.CanChat(nil) {
		return
	}
	b := s.resolveActiveBuild()
	if b == nil || m.ReasoningLevelsBuild == b.ID {
		return
	}
	if !s.levels.start(m.ID) {
		return
	}
	go func() {
		defer s.levels.done(m.ID)
		ctx, cancel := context.WithTimeout(context.Background(), reasoningLevelsTimeout)
		defer cancel()
		levels, known, err := s.readReasoningLevels(ctx, s.registry.RouterName(m.ID))
		if err != nil {
			slog.Debug("could not read the reasoning effort levels", "model", m.PublicName(), "error", err)
			return // read again on the next load
		}
		if err := s.registry.SetReasoningLevels(m.ID, levels, known, b.ID); err != nil {
			slog.Warn("failed to save the reasoning effort levels", "model", m.PublicName(), "error", err)
			return
		}
		if known {
			slog.Info("read the reasoning effort levels", "model", m.PublicName(), "levels", levels, "build", b.ID)
		}
	}()
}

// readReasoningLevels asks llama-server which reasoning_effort values the
// model's chat template accepts: /props of the loaded model,
// chat_template_caps.reasoning_effort_levels. autoload=false, so a model
// unloaded in the meantime is not loaded again. known is false when the
// build does not report the levels (older than llama.cpp #31271); an
// error means no answer, to be tried again.
func (s *Server) readReasoningLevels(ctx context.Context, routerName string) (levels []string, known bool, err error) {
	u := fmt.Sprintf("http://localhost:%d/props?model=%s&autoload=false", s.llamaPort(), url.QueryEscape(routerName))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, false, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("/props answered %s", resp.Status)
	}
	var props struct {
		Caps map[string]json.RawMessage `json:"chat_template_caps"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&props); err != nil {
		return nil, false, err
	}
	if props.Caps == nil {
		return nil, false, fmt.Errorf("/props has no chat_template_caps")
	}
	raw, ok := props.Caps["reasoning_effort_levels"]
	if !ok {
		return nil, false, nil
	}
	if err := json.Unmarshal(raw, &levels); err != nil {
		return nil, false, err
	}
	if levels == nil {
		levels = []string{}
	}
	return levels, true, nil
}
