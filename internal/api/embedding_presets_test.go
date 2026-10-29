package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The "Get embedding models" card lists every curated model with its
// context, licence and size, and marks the ones already downloaded.
func TestEmbeddingPresetsCard(t *testing.T) {
	s := newTestServer(t)
	if err := s.registry.Add(&models.Model{ID: "q", ModelID: "Qwen/Qwen3-Embedding-0.6B-GGUF",
		Filename: "Qwen3-Embedding-0.6B-Q8_0.gguf"}); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/models/embedding-presets", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleEmbeddingPresets(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	body := rec.Body.String()
	for _, p := range models.CuratedEmbeddingModels() {
		if !strings.Contains(body, p.Name) {
			t.Errorf("card does not list %s", p.Name)
		}
	}
	for _, want := range []string{"262144", "Apache-2.0", "Gemma", "4.7 GB", "310 MB", "Downloaded"} {
		if !strings.Contains(body, want) {
			t.Errorf("card does not show %q", want)
		}
	}
	if n := strings.Count(body, "Downloaded"); n != 1 {
		t.Errorf("%d models marked downloaded, want 1", n)
	}
}

// An embedding model's settings show the context size, which is now
// passed to llama-server along with a micro-batch as large.
func TestEmbeddingConfigPanelShowsContext(t *testing.T) {
	s := newTestServer(t)
	m := &models.Model{ID: "e", ModelID: "ggml-org/embeddinggemma-2-GGUF", Filename: "e.gguf",
		ContextLength: 262144, PoolingType: models.PoolingMean, Embedding: true}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	data, err := s.configPanelData("e")
	if err != nil {
		t.Fatal(err)
	}
	if !data.IsEmbedding {
		t.Error("IsEmbedding is false for a model whose file says it is one")
	}
	if !strings.Contains(data.EffectiveFlags, "--ctx-size 8192") || !strings.Contains(data.EffectiveFlags, "--ubatch-size 8192") {
		t.Errorf("effective flags = %q, want the default 8192 context and micro-batch", data.EffectiveFlags)
	}
	rec := httptest.NewRecorder()
	s.renderPartial(rec, "model_config", data)
	body := rec.Body.String()
	if data.DefaultContext != models.EmbeddingDefaultContext {
		t.Errorf("DefaultContext = %d, want %d", data.DefaultContext, models.EmbeddingDefaultContext)
	}
	for _, want := range []string{`name="context_size"`, "one embedding request can hold", "Model Default (8192)"} {
		if !strings.Contains(body, want) {
			t.Errorf("config panel does not contain %q", want)
		}
	}
}

// An embedding model with no GPU assignment is put on the GPU with the
// most room and the preset follows. A chat model, a helper model and an
// assignment someone chose are left alone.
func TestPlaceEmbeddingModel(t *testing.T) {
	s := newTestServer(t)
	hw := models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, VRAMTotalMiB: 16376, OtherUsedMiB: 2000},
		{Index: 1, VRAMTotalMiB: 16376},
		{Index: 2, VRAMTotalMiB: 16376, ReservedMiB: 6000},
	}}
	emb := func(id string) *models.Model {
		return &models.Model{ID: id, ModelID: "org/" + id, Filename: id + ".gguf", FilePath: "/models/" + id + ".gguf",
			Embedding: true, PoolingType: models.PoolingMean, NLayers: 24, NEmbd: 768, NHead: 3, NKVHead: 1,
			ContextLength: 2048, SizeBytes: 300 << 20}
	}
	chat := &models.Model{ID: "chat", ModelID: "org/Chat-GGUF", Filename: "c.gguf", NLayers: 32, NEmbd: 4096,
		NHead: 32, NKVHead: 8, ContextLength: 32768, SizeBytes: 5 << 30}
	helper := emb("helper")
	helper.HelperRole = true
	for _, m := range []*models.Model{emb("e"), emb("chosen"), chat, helper} {
		if err := s.registry.Add(m); err != nil {
			t.Fatal(err)
		}
	}
	cfg, _ := s.registry.GetConfig("chosen")
	cfg.GPUAssign = "all"
	if err := s.registry.SetConfig("chosen", cfg); err != nil {
		t.Fatal(err)
	}

	for _, id := range []string{"e", "chosen", "chat", "helper"} {
		s.placeEmbeddingModel(id, hw)
	}
	want := map[string]string{"e": "1", "chosen": "all", "chat": "", "helper": ""}
	for id, w := range want {
		if cfg, _ := s.registry.GetConfig(id); cfg.GPUAssign != w {
			t.Errorf("%s: GPU assignment %q, want %q", id, cfg.GPUAssign, w)
		}
	}

	m, _ := s.registry.Get("e")
	cfg, _ = s.registry.GetConfig("e")
	ini := models.GeneratePresetINI("/models", []*models.Model{m}, map[string]*models.ModelConfig{"e": cfg}, models.Target{})
	if !strings.Contains(ini, "tensor-split = 0,1,0") {
		t.Errorf("preset does not keep the model on GPU 1:\n%s", ini)
	}
}
