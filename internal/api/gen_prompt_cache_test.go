package api

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/routerclient"
)

// With the setting on, an eviction unload saves each slot holding a long
// enough conversation, and the next load restores those slots and
// deletes the files. Short conversations are not saved.
func TestPromptCacheSavedAndRestored(t *testing.T) {
	s, f, m := startFakeRouter(t)
	store, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = store
	s.cfg.GenerationEnabled, s.cfg.GenEvict, s.cfg.GenKeepPromptCache = true, true, true
	if kv := s.promptCacheRouterEnv(); kv != "LLAMA_ARG_SLOT_SAVE_PATH="+s.promptCacheDir() {
		t.Fatalf("router env = %q", kv)
	}
	f.mu.Lock()
	f.slots = []routerclient.Slot{{ID: 0, PromptTokens: 26000}, {ID: 1, PromptTokens: 300}, {ID: 2}}
	f.mu.Unlock()
	name := s.registry.RouterName(m.ID)

	s.savePromptCache(context.Background(), m, name)
	file := promptCacheFile(m.ID, 0)
	if got := strings.Join(f.slotCalls, "; "); got != "save 0 "+file+" "+name {
		t.Fatalf("slot calls = %q, want only slot 0 saved", got)
	}
	// The file is the model server's to write; stand in for it.
	path := filepath.Join(s.promptCacheDir(), file)
	os.WriteFile(path, []byte("kv"), 0o644)

	s.restorePromptCache(context.Background(), m, name)
	if f.slotCalls[1] != "restore 0 "+file+" "+name {
		t.Errorf("slot calls = %q, want the restore of slot 0", f.slotCalls)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("the file was kept after the restore")
	}
	s.restorePromptCache(context.Background(), m, name)
	if len(f.slotCalls) != 2 {
		t.Errorf("a second load restored again: %q", f.slotCalls)
	}

	// Off, nothing is saved and the router is not given the directory.
	s.cfg.GenKeepPromptCache = false
	s.savePromptCache(context.Background(), m, name)
	if len(f.slotCalls) != 2 || s.promptCacheRouterEnv() != "" {
		t.Errorf("with the setting off: calls %q", f.slotCalls)
	}
}
