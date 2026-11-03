package api

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// addRoomModel registers an enabled model of sizeGiB on disk.
func addRoomModel(t *testing.T, s *Server, id string, sizeGiB int64) *models.Model {
	t.Helper()
	m := &models.Model{
		ID: id, ModelID: "org/" + id + "-GGUF", Quant: "Q4_K_M",
		Filename: id + ".gguf", FilePath: filepath.Join(t.TempDir(), id+".gguf"),
		SizeBytes: sizeGiB << 30,
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig(m.ID, &models.ModelConfig{Enabled: true}); err != nil {
		t.Fatal(err)
	}
	return m
}

// tryEvict never waits: it fails at once for a model with a request in
// progress, and an unload it starts holds new requests back until done.
func TestLLMUseTryEvict(t *testing.T) {
	var u llmUse
	ctx := context.Background()
	end, err := u.begin(ctx, "qwen")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := u.tryEvict("qwen"); ok {
		t.Fatal("tryEvict took a model with a request in progress")
	}
	end()

	done, ok := u.tryEvict("qwen")
	if !ok {
		t.Fatal("tryEvict refused an idle model")
	}
	if _, ok := u.tryEvict("qwen"); ok {
		t.Error("a second tryEvict took a model already being unloaded")
	}
	if !u.busy("qwen") {
		t.Error("a model being unloaded is not busy")
	}
	began := make(chan struct{})
	go func() {
		end, _ := u.begin(ctx, "qwen")
		end()
		close(began)
	}()
	select {
	case <-began:
		t.Fatal("a request began while the model was being unloaded")
	case <-time.After(30 * time.Millisecond):
	}
	done()
	<-began
}

// The idle list has the loaded models no request uses, least recently
// used first, each with its planned memory; a busy model, one still
// loading, an unknown one and except are left out.
func TestIdleLLMsByLRU(t *testing.T) {
	s, f, _ := startFakeRouter(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{{Index: 0, Name: "R9700", VRAMTotalMiB: 32624, OtherUsedMiB: 1300}}}
	for _, id := range []string{"chat", "embed", "rerank", "busy", "loading", "except"} {
		addRoomModel(t, s, id, 2)
	}
	f.reset(map[string]string{
		"chat": "loaded", "embed": "loaded", "rerank": "loaded", "busy": "loaded",
		"loading": "loading", "except": "loaded", "stranger": "loaded",
	}, nil)

	s.llmUse.touch("rerank")
	time.Sleep(2 * time.Millisecond)
	s.llmUse.touch("chat")
	end, err := s.llmUse.begin(context.Background(), "busy")
	if err != nil {
		t.Fatal(err)
	}
	defer end()

	got := s.idleLLMsByLRU("except")
	var ids []string
	for _, l := range got {
		ids = append(ids, l.id)
		if l.mib[0] <= 0 {
			t.Errorf("%s has no planned memory on GPU 0: %v", l.id, l.mib)
		}
	}
	// embed was never used, so it sorts first.
	want := []string{"embed", "rerank", "chat"}
	if len(ids) != len(want) {
		t.Fatalf("idle = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("idle = %v, want %v", ids, want)
		}
	}

	// loaded and loading models of ours count; stranger is not ours.
	if n := s.loadedLLMCount(); n != 6 {
		t.Errorf("loadedLLMCount = %d, want 6", n)
	}
}
