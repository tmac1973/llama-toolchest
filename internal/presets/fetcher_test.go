package presets

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func f64(v float64) *float64 { return &v }
func i(v int) *int           { return &v }

// newTestFetcher wires a Fetcher against two httptest servers standing in for
// huggingface.co and unsloth.ai/docs. Handlers not registered 404, which the
// fetcher must treat as a silent miss.
func newTestFetcher(t *testing.T, hf, docs map[string]string) *Fetcher {
	t.Helper()
	mkServer := func(routes map[string]string) *httptest.Server {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if body, ok := routes[r.URL.Path]; ok {
				w.Write([]byte(body))
				return
			}
			http.NotFound(w, r)
		}))
		t.Cleanup(srv.Close)
		return srv
	}
	hfSrv := mkServer(hf)
	docsSrv := mkServer(docs)
	return &Fetcher{
		HFBase:   hfSrv.URL,
		DocsBase: docsSrv.URL + "/docs",
		Client:   hfSrv.Client(),
		CacheDir: t.TempDir(),
	}
}

func TestFetchChainMergesAllSources(t *testing.T) {
	docs := map[string]string{
		"/docs/llms.txt":          "- [Qwen 3.8: How to Run](/docs/models/qwen3.8.md): d\n",
		"/docs/models/qwen3.8.md": qwen38PageMD,
	}
	hf := map[string]string{
		"/Qwen/Qwen3.8-27B/raw/main/generation_config.json": `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05, "eos_token_id": 1}`,
	}
	f := newTestFetcher(t, hf, docs)

	// Model as it looks after GGUF parsing: embedded default + base repo.
	m := &models.Model{
		ID:            "unsloth--Qwen3.8-27B-GGUF--q4",
		ModelID:       "unsloth/Qwen3.8-27B-GGUF",
		BaseModelRepo: "Qwen/Qwen3.8-27B",
		SamplingPresets: []models.SamplingPreset{{
			Name: "default", Label: "Model-embedded default", Source: "gguf",
			Temperature: f64(1.0), TopP: f64(0.95), TopK: i(20),
		}},
	}

	out := f.Fetch(context.Background(), m)
	byName := map[string]models.SamplingPreset{}
	for _, p := range out {
		byName[p.Name] = p
	}
	if len(out) != 3 {
		t.Fatalf("got %d presets %v, want default+thinking+non-thinking", len(out), byName)
	}

	// The gguf default keeps its provenance and values; gen_config only
	// back-fills fields the gguf entry lacked (repeat_penalty).
	def := byName["default"]
	if def.Source != "gguf" {
		t.Errorf("default source = %q, want gguf (gen_config must not take over)", def.Source)
	}
	if def.RepeatPenalty == nil || *def.RepeatPenalty != 1.05 {
		t.Errorf("default repeat_penalty = %v, want 1.05 back-filled from generation_config", def.RepeatPenalty)
	}
	// Docs variants arrive intact.
	if nt := byName["non-thinking"]; nt.PresencePenalty == nil || *nt.PresencePenalty != 1.5 {
		t.Errorf("non-thinking presence_penalty = %v, want 1.5", byName["non-thinking"].PresencePenalty)
	}
	// Order: default first.
	if out[0].Name != "default" {
		t.Errorf("first preset = %q, want default", out[0].Name)
	}
	// Input model untouched (Fetch must clone).
	if len(m.SamplingPresets) != 1 {
		t.Errorf("input presets mutated: %+v", m.SamplingPresets)
	}
}

func TestFetchAllMissesReturnsInput(t *testing.T) {
	f := newTestFetcher(t, map[string]string{}, map[string]string{})
	m := &models.Model{ModelID: "org/Unknown-GGUF"}
	if out := f.Fetch(context.Background(), m); len(out) != 0 {
		t.Errorf("expected no presets on total miss, got %+v", out)
	}
}

func TestFetchResolvesBaseRepoFromTags(t *testing.T) {
	hf := map[string]string{
		"/api/models/bartowski/Old-Model-GGUF":            `{"tags":["gguf","base_model:acme/Old-Model","base_model:quantized:acme/Old-Model"]}`,
		"/acme/Old-Model/raw/main/generation_config.json": `{"temperature": 0.6, "top_p": 0.9}`,
	}
	f := newTestFetcher(t, hf, map[string]string{})
	m := &models.Model{ModelID: "bartowski/Old-Model-GGUF"} // no BaseModelRepo: pre-Nov-2025 quant
	out := f.Fetch(context.Background(), m)
	if len(out) != 1 || out[0].Source != "generation_config.json" {
		t.Fatalf("got %+v, want one generation_config preset", out)
	}
	if out[0].Temperature == nil || *out[0].Temperature != 0.6 {
		t.Errorf("temperature = %v, want 0.6", out[0].Temperature)
	}
}

func TestFetchGenConfigGatedFallsBackToMirror(t *testing.T) {
	hf := map[string]string{
		// meta-llama is gated (404s here); the unsloth mirror serves it.
		"/unsloth/Gated-Model/raw/main/generation_config.json": `{"temperature": 0.6, "top_p": 0.9}`,
	}
	f := newTestFetcher(t, hf, map[string]string{})
	p := f.fetchGenerationConfig(context.Background(), "meta-llama/Gated-Model")
	if p == nil || p.Temperature == nil || *p.Temperature != 0.6 {
		t.Fatalf("mirror fallback failed: %+v", p)
	}
}

func TestFetchGenConfigTokenIDsOnlyIsAMiss(t *testing.T) {
	hf := map[string]string{
		"/z/M/raw/main/generation_config.json": `{"bos_token_id": 1, "eos_token_id": 2}`,
	}
	f := newTestFetcher(t, hf, map[string]string{})
	if p := f.fetchGenerationConfig(context.Background(), "z/M"); p != nil {
		t.Errorf("token-IDs-only config must be a miss, got %+v", p)
	}
}

func TestFetchParamsFile(t *testing.T) {
	hf := map[string]string{
		"/unsloth/Q-GGUF/raw/main/params": `{"stop": ["a"], "temperature": 0.7, "min_p": 0.0, "repeat_penalty": 1.05, "top_k": 20, "top_p": 0.8}`,
	}
	f := newTestFetcher(t, hf, map[string]string{})
	p := f.fetchParamsFile(context.Background(), "unsloth/Q-GGUF")
	if p == nil {
		t.Fatal("params file not parsed")
	}
	if p.MinP == nil || *p.MinP != 0.0 || p.RepeatPenalty == nil || *p.RepeatPenalty != 1.05 || p.TopK == nil || *p.TopK != 20 {
		t.Errorf("params fields wrong: %+v", p)
	}
}
