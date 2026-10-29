package models

import (
	"reflect"
	"strings"
	"testing"
)

// A Qwen3-Embedding file is a qwen3, the same architecture as the chat
// models; only its pooling type says it is an embedding model. The parse
// must read the key and the record must carry the result.
func TestParseGGUFPoolingType(t *testing.T) {
	path := writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "qwen3"},
		{"qwen3.context_length", ggufTypeUint32, uint32(32768)},
		{"qwen3.pooling_type", ggufTypeUint32, uint32(PoolingLast)},
		{"qwen3.block_count", ggufTypeUint32, uint32(28)},
	})
	meta, err := ParseGGUFMeta(path)
	if err != nil {
		t.Fatal(err)
	}
	if meta.PoolingType != PoolingLast {
		t.Errorf("PoolingType = %d, want %d", meta.PoolingType, PoolingLast)
	}
	if meta.NLayers != 28 {
		t.Errorf("NLayers = %d, want 28 (parser misaligned after pooling_type?)", meta.NLayers)
	}
	m := &Model{ID: "org--some-model", ModelID: "org/some-model"}
	meta.ApplyTo(m)
	if !m.Embedding || m.PoolingType != PoolingLast {
		t.Errorf("after ApplyTo: Embedding %v, PoolingType %d; want true, %d", m.Embedding, m.PoolingType, PoolingLast)
	}
	if !m.IsEmbedding() {
		t.Error("a model whose file says it is an embedding model is not one by IsEmbedding")
	}
}

func TestGGUFMetaIsEmbedding(t *testing.T) {
	tests := []struct {
		arch    string
		pooling int
		want    bool
	}{
		{"qwen3", PoolingNone, false},   // a chat model
		{"qwen3", PoolingLast, true},    // Qwen3-Embedding
		{"llama", PoolingMean, true},    // any architecture with a pooling type
		{"bert", PoolingCLS, true},      // bge-m3
		{"bert", PoolingNone, true},     // an older conversion without the key
		{"gemma-embedding2", 0, true},   // EmbeddingGemma 2
		{"gemma-embedding", 0, true},    // EmbeddingGemma
		{"bert", PoolingRank, false},    // a reranker
		{"qwen3", PoolingRank, false},   // a causal reranker
		{"gemma4", PoolingNone, false},  // not an encoder
		{"nomic-bert-moe", 1, true},     // nomic v2
		{"eurobert", PoolingNone, true}, // Jina v5 nano
	}
	for _, tt := range tests {
		meta := &GGUFMeta{Architecture: tt.arch, PoolingType: tt.pooling}
		if got := meta.IsEmbedding(); got != tt.want {
			t.Errorf("arch %s, pooling %d: IsEmbedding = %v, want %v", tt.arch, tt.pooling, got, tt.want)
		}
	}
}

// The name is all there is to go on for a download in progress, so every
// curated repo must match it, and chat models must not.
func TestIsEmbeddingModelNames(t *testing.T) {
	embedding := []string{
		"Qwen/Qwen3-Embedding-0.6B-GGUF",
		"ggml-org/embeddinggemma-2-GGUF",
		"ggml-org/embeddinggemma-300M-GGUF",
		"ibm-granite/granite-embedding-278m-multilingual",
		"nomic-ai/nomic-embed-text-v2-moe-GGUF",
		"gpustack/bge-m3-GGUF",
		"jinaai/jina-embeddings-v4-text-retrieval-GGUF",
	}
	for _, n := range embedding {
		if !IsEmbeddingModel(n) {
			t.Errorf("IsEmbeddingModel(%q) = false, want true", n)
		}
	}
	chat := []string{
		"unsloth/Qwen3.5-27B-GGUF",
		"bartowski/Llama-3.3-70B-Instruct-GGUF",
		"unsloth/gemma-4-E4B-it-GGUF",
		"ggml-org/gpt-oss-20b-GGUF",
	}
	for _, n := range chat {
		if IsEmbeddingModel(n) {
			t.Errorf("IsEmbeddingModel(%q) = true, want false", n)
		}
	}
	for _, p := range CuratedEmbeddingModels() {
		if !IsEmbeddingModel(p.Repo) {
			t.Errorf("curated %s: repo %q does not look like an embedding model by name", p.Name, p.Repo)
		}
	}
}

func TestCuratedEmbeddingModelsComplete(t *testing.T) {
	repos := map[string]bool{}
	for _, p := range CuratedEmbeddingModels() {
		if p.Name == "" || p.Repo == "" || p.Filename == "" || p.SizeMB <= 0 || p.Dims <= 0 ||
			p.Context <= 0 || p.License == "" || p.Desc == "" {
			t.Errorf("curated %q has an empty field: %+v", p.Name, p)
		}
		if !strings.HasSuffix(strings.ToLower(p.Filename), ".gguf") {
			t.Errorf("curated %s: filename %q is not a GGUF file", p.Name, p.Filename)
		}
		if repos[p.Repo] {
			t.Errorf("curated repo %s is listed twice", p.Repo)
		}
		repos[p.Repo] = true
	}
}

func TestEmbeddingPresetSizeLabel(t *testing.T) {
	if got := (EmbeddingModelPreset{SizeMB: 310}).SizeLabel(); got != "310 MB" {
		t.Errorf("310 MB: got %q", got)
	}
	if got := (EmbeddingModelPreset{SizeMB: 4677}).SizeLabel(); got != "4.7 GB" {
		t.Errorf("4677 MB: got %q", got)
	}
}

func TestEmbeddingLaunchConfig(t *testing.T) {
	mean := &Model{ContextLength: 2048, PoolingType: PoolingMean}
	last := &Model{ContextLength: 40960, PoolingType: PoolingLast}
	tests := []struct {
		name           string
		m              *Model
		cfg            ModelConfig
		ctx, ub, batch int
	}{
		{"capped at the trained context", mean, ModelConfig{ContextSize: 8192}, 2048, 2048, 0},
		{"model default is the trained context", mean, ModelConfig{}, 2048, 2048, 0},
		{"batch raised past its default", &Model{ContextLength: 8192}, ModelConfig{ContextSize: 8192}, 8192, 8192, 8192},
		{"below the trained context kept", mean, ModelConfig{ContextSize: 1024}, 1024, 1024, 0},
		{"a set micro-batch is kept", mean, ModelConfig{ContextSize: 2048, UBatchSize: 256}, 2048, 256, 0},
		{"a larger batch is kept", mean, ModelConfig{ContextSize: 2048, BatchSize: 4096}, 2048, 2048, 4096},
		{"last-token pooling splits the input", last, ModelConfig{ContextSize: 8192}, 8192, 0, 0},
		{"model default stops at 8192", &Model{ContextLength: 262144, PoolingType: PoolingMean},
			ModelConfig{}, EmbeddingDefaultContext, EmbeddingDefaultContext, EmbeddingDefaultContext},
		{"model default with last-token pooling is the trained context", last, ModelConfig{}, 40960, 0, 0},
		{"a context set on purpose is followed", &Model{ContextLength: 262144, PoolingType: PoolingMean},
			ModelConfig{ContextSize: 16384}, 16384, 16384, 16384},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.cfg
			got := EmbeddingLaunchConfig(&cfg, tt.m)
			if got.ContextSize != tt.ctx || got.UBatchSize != tt.ub || got.BatchSize != tt.batch {
				t.Errorf("ctx %d, ub %d, batch %d; want %d, %d, %d",
					got.ContextSize, got.UBatchSize, got.BatchSize, tt.ctx, tt.ub, tt.batch)
			}
			if err := got.ValidateBatchSizes(); err != nil {
				t.Errorf("launch config is not valid: %v", err)
			}
			if !reflect.DeepEqual(cfg, tt.cfg) {
				t.Errorf("the saved config was changed: %+v", cfg)
			}
		})
	}

	// Nothing to change: the same config comes back, as MoESplitConfig does.
	cfg := &ModelConfig{ContextSize: 8192}
	if got := EmbeddingLaunchConfig(cfg, last); got != cfg {
		t.Error("an unchanged config should be returned as is")
	}
	// A model with no known context and none set: nothing to give it.
	if got := EmbeddingLaunchConfig(&ModelConfig{}, &Model{}); got.ContextSize != 0 || got.UBatchSize != 0 {
		t.Errorf("no context anywhere: got ctx %d, ub %d", got.ContextSize, got.UBatchSize)
	}
}

// The preset file must give an embedding model its context and the
// micro-batch that lets an input that long through. Before, it gave no
// context at all, and llama-server's 512-token micro-batch refused any
// longer input.
func TestPresetEmbeddingContext(t *testing.T) {
	m := &Model{ID: "e", ModelID: "ggml-org/embeddinggemma-2-GGUF", Filename: "e.gguf", FilePath: "/models/e.gguf",
		ContextLength: 262144, PoolingType: PoolingMean, Embedding: true}
	cfg := &ModelConfig{Enabled: true, ContextSize: 8192, GPULayers: 999}
	ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"e": cfg}, Target{})
	for _, want := range []string{"embeddings = true", "ctx-size = 8192", "ubatch-size = 8192", "batch-size = 8192"} {
		if !strings.Contains(ini, want) {
			t.Errorf("want %q in preset:\n%s", want, ini)
		}
	}
	if cfg.UBatchSize != 0 || cfg.BatchSize != 0 {
		t.Errorf("the saved config was changed: ub %d, batch %d", cfg.UBatchSize, cfg.BatchSize)
	}
}

// The estimate follows what llama-server is given: an embedding model's
// context is capped at its own, so a saved 8192 on a 512-token model is
// estimated as 512.
func TestVRAMEmbeddingUsesLaunchConfig(t *testing.T) {
	m := &Model{ID: "e", ModelID: "org/bge-small-GGUF", Embedding: true, PoolingType: PoolingCLS,
		NLayers: 12, NEmbd: 384, NHead: 12, NKVHead: 12, ContextLength: 512, SizeBytes: 130 << 20}
	saved := VRAMBreakdownForConfigOn(m, &ModelConfig{ContextSize: 8192, GPULayers: 999}, 1)
	launched := VRAMBreakdownForConfigOn(m, &ModelConfig{ContextSize: 512, UBatchSize: 512, GPULayers: 999}, 1)
	if saved != launched {
		t.Errorf("estimate for the saved config %+v, for the launch config %+v; want the same", saved, launched)
	}
}
