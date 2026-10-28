package api

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/routerclient"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
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

// A request's prompt cache is restored after the request is counted, so
// an unload waits for the restore. A restore reaching the router after an
// unload made it load the model again behind every room check (seen on
// compute2: the 27B came back between an unload and ace starting).
func TestPromptCacheRestoredWhileCounted(t *testing.T) {
	s, f, m := startFakeRouter(t)
	store, err := generation.Load(s.cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = store
	s.cfg.GenerationEnabled, s.cfg.GenEvict, s.cfg.GenKeepPromptCache = true, true, true
	name := s.registry.RouterName(m.ID)
	f.mu.Lock()
	f.slots = []routerclient.Slot{{ID: 0, PromptTokens: 26000}}
	f.status = map[string]string{name: "loaded"}
	f.mu.Unlock()
	s.savePromptCache(context.Background(), m, name)
	os.WriteFile(filepath.Join(s.promptCacheDir(), promptCacheFile(m.ID, 0)), []byte("kv"), 0o644)

	gate := make(chan struct{})
	f.mu.Lock()
	f.restoreGate = gate
	f.mu.Unlock()
	done := make(chan func(), 1)
	go func() {
		end, err := s.loadAndCountRequest(context.Background(), []byte(`{"model":"`+m.ID+`"}`))
		if err != nil {
			t.Error(err)
		}
		done <- end
	}()
	testutil.Eventually(t, func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return len(f.slotCalls) == 2
	}, "the cache was not restored")

	// The restore is under way: an unload waits for it.
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := s.llmUse.evict(ctx, m.ID); err == nil {
		t.Error("an unload went ahead during the restore")
	}
	close(gate)
	(<-done)()
}
