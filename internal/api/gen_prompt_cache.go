package api

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Keeping the prompt cache across an eviction (Settings → Generation
// backends, config.GenKeepPromptCache). Before unloadForGen unloads an
// LLM, each of its slots that holds a conversation is saved to a file
// through llama-server's /slots API; when the model is loaded again for
// a request, the files are restored before the request goes on, so the
// reply processes only the new message instead of the whole
// conversation. Measured on 3×A4000 with Qwen3.6-35B-A3B and a
// 26,600-token conversation: 21.4 s to process it again, 0.5 s to save
// (609 MB) and 0.2 s to restore.
//
// The router hands LLAMA_ARG_SLOT_SAVE_PATH to every model's server
// (startRouter sets it), which is what turns the /slots actions on.

// promptCacheMinTokens is the shortest conversation worth a file: a
// shorter one is processed again in about a second.
const promptCacheMinTokens = 1024

// promptCacheDirName is the directory under the data directory that
// holds the saved files.
const promptCacheDirName = "slot-cache"

// promptCacheStore records the files saved for each model (by registry
// ID) until they are restored.
type promptCacheStore struct {
	mu    sync.Mutex
	saved map[string][]savedSlot
}

// savedSlot is one slot's saved prompt cache.
type savedSlot struct {
	id     int
	file   string
	tokens int
}

func (p *promptCacheStore) put(id string, slots []savedSlot) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.saved == nil {
		p.saved = map[string][]savedSlot{}
	}
	p.saved[id] = slots
}

// take removes and returns what was saved for model id.
func (p *promptCacheStore) take(id string) []savedSlot {
	p.mu.Lock()
	defer p.mu.Unlock()
	slots := p.saved[id]
	delete(p.saved, id)
	return slots
}

// reset forgets every saved file.
func (p *promptCacheStore) reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.saved = nil
}

// promptCacheOn reports whether LLM prompt caches are saved across
// evictions.
func (s *Server) promptCacheOn() bool {
	return s.genEvictOn() && s.cfgCopy().GenKeepPromptCache
}

func (s *Server) promptCacheDir() string {
	return filepath.Join(s.cfg.DataDir, promptCacheDirName)
}

// promptCacheRouterEnv prepares the directory for a router start and
// returns the variable that turns slot saving on, or "" when it is off.
// Files from an earlier run are deleted: the models they belong to are
// not loaded any more, and nothing records them.
func (s *Server) promptCacheRouterEnv() string {
	s.promptCache.reset()
	dir := s.promptCacheDir()
	os.RemoveAll(dir)
	if !s.promptCacheOn() {
		return ""
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Warn("prompt cache directory not created; prompt caches will not be kept", "dir", dir, "error", err)
		return ""
	}
	return "LLAMA_ARG_SLOT_SAVE_PATH=" + dir
}

// promptCacheFile is the file name for slot id of model id: a short hash
// of the registry ID, which can be long and holds characters that a
// file name check could refuse.
func promptCacheFile(modelID string, slot int) string {
	sum := sha256.Sum256([]byte(modelID))
	return fmt.Sprintf("%s-slot%d.bin", hex.EncodeToString(sum[:6]), slot)
}

// savePromptCache saves the prompt cache of each of m's slots holding a
// conversation of promptCacheMinTokens or more. A failure only means the
// next reply processes the conversation again, so it is logged, not
// returned.
func (s *Server) savePromptCache(ctx context.Context, m *models.Model, routerName string) {
	if !s.promptCacheOn() {
		return
	}
	slots, err := s.process.Slots(ctx, routerName)
	if err != nil {
		slog.Warn("prompt cache not saved: could not list the model's slots", "model", m.PublicName(), "error", err)
		return
	}
	var saved []savedSlot
	for _, sl := range slots {
		if sl.PromptTokens < promptCacheMinTokens {
			continue
		}
		file := promptCacheFile(m.ID, sl.ID)
		start := time.Now()
		res, err := s.process.SaveSlot(ctx, routerName, sl.ID, file)
		if err != nil {
			// A router started before the setting was switched on
			// answers that slot actions are not supported.
			slog.Warn("prompt cache not saved", "model", m.PublicName(), "slot", sl.ID, "error", err)
			continue
		}
		slog.Info("saved prompt cache before unloading", "model", m.PublicName(), "slot", sl.ID,
			"tokens", res.Tokens, "mib", res.Bytes>>20, "took", time.Since(start).Round(time.Millisecond))
		saved = append(saved, savedSlot{id: sl.ID, file: file, tokens: res.Tokens})
	}
	if len(saved) > 0 {
		s.promptCache.put(m.ID, saved)
	}
}

// restorePromptCache restores what savePromptCache saved for m, now
// that m is loaded again, and deletes the files. A slot that cannot be
// restored (the model's settings changed meanwhile, say) is skipped: its
// conversation is processed again.
func (s *Server) restorePromptCache(ctx context.Context, m *models.Model, routerName string) {
	for _, sl := range s.promptCache.take(m.ID) {
		start := time.Now()
		res, err := s.process.RestoreSlot(ctx, routerName, sl.id, sl.file)
		if err != nil {
			slog.Warn("prompt cache not restored", "model", m.PublicName(), "slot", sl.id, "error", err)
		} else {
			slog.Info("restored prompt cache after loading", "model", m.PublicName(), "slot", sl.id,
				"tokens", res.Tokens, "took", time.Since(start).Round(time.Millisecond))
		}
		os.Remove(filepath.Join(s.promptCacheDir(), sl.file))
	}
}
