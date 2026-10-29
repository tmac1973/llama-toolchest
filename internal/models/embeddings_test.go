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

// Three A4000-sized cards, the way compute2 has them.
func threeCards() Hardware {
	return Hardware{GPUs: []GPUSpec{
		{Index: 0, VRAMTotalMiB: 16376}, {Index: 1, VRAMTotalMiB: 16376}, {Index: 2, VRAMTotalMiB: 16376},
	}}
}

func eg2() *Model {
	return &Model{ID: "e", ModelID: "ggml-org/embeddinggemma-2-GGUF", Arch: "gemma-embedding2", Embedding: true,
		PoolingType: PoolingMean, NLayers: 24, NEmbd: 512, NHead: 4, NKVHead: 1, ContextLength: 262144,
		SizeBytes: 310 << 20}
}

func TestEmbeddingGPU(t *testing.T) {
	cfg := DefaultConfig()

	t.Run("most room", func(t *testing.T) {
		hw := threeCards()
		hw.GPUs[0].OtherUsedMiB = 2000 // a desktop
		hw.GPUs[2].ReservedMiB = 6000  // a generation backend
		opt, ok := EmbeddingGPU(eg2(), cfg, hw)
		if !ok || opt.Value != "1" {
			t.Errorf("got %q, %v; want GPU 1, the one with the most room", opt.Value, ok)
		}
	})
	t.Run("ties go to the lower index", func(t *testing.T) {
		opt, ok := EmbeddingGPU(eg2(), cfg, threeCards())
		if !ok || opt.Value != "0" {
			t.Errorf("got %q, %v; want GPU 0", opt.Value, ok)
		}
	})
	t.Run("the integrated GPU is never chosen", func(t *testing.T) {
		hw := Hardware{GPUs: []GPUSpec{
			{Index: 0, VRAMTotalMiB: 65536, IsIGPU: true}, {Index: 1, VRAMTotalMiB: 16376}, {Index: 2, VRAMTotalMiB: 8192},
		}}
		opt, ok := EmbeddingGPU(eg2(), cfg, hw)
		if !ok || opt.Value != "1" {
			t.Errorf("got %q, %v; want GPU 1, the larger discrete GPU", opt.Value, ok)
		}
	})
	t.Run("one discrete GPU changes nothing", func(t *testing.T) {
		hw := Hardware{GPUs: []GPUSpec{{Index: 0, VRAMTotalMiB: 16376, IsIGPU: true}, {Index: 1, VRAMTotalMiB: 16376}}}
		if opt, ok := EmbeddingGPU(eg2(), cfg, hw); ok {
			t.Errorf("got %q; want no placement with one discrete GPU", opt.Value)
		}
		if opt, ok := EmbeddingGPU(eg2(), cfg, Hardware{GPUs: threeCards().GPUs[:1]}); ok {
			t.Errorf("got %q; want no placement with one GPU", opt.Value)
		}
	})
	t.Run("too large for one GPU", func(t *testing.T) {
		big := &Model{ID: "q", ModelID: "Qwen/Qwen3-Embedding-8B-GGUF", Arch: "qwen3", Embedding: true,
			PoolingType: PoolingLast, NLayers: 36, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 40960,
			SizeBytes: 4677 << 20}
		hw := Hardware{GPUs: []GPUSpec{{Index: 0, VRAMTotalMiB: 4096}, {Index: 1, VRAMTotalMiB: 4096}}}
		if opt, ok := EmbeddingGPU(big, cfg, hw); ok {
			t.Errorf("got %q; want no placement when the model fits on no GPU alone", opt.Value)
		}
	})
}

func TestRegistryPlaceOnGPU(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir, dir)
	for _, id := range []string{"new", "chosen"} {
		if err := r.Add(&Model{ID: id, ModelID: "org/" + id}); err != nil {
			t.Fatal(err)
		}
	}
	chosen, _ := r.GetConfig("chosen")
	chosen.GPUAssign = "all"
	if err := r.SetConfig("chosen", chosen); err != nil {
		t.Fatal(err)
	}

	if placed, err := r.PlaceOnGPU("new", "1", 3); err != nil || !placed {
		t.Fatalf("PlaceOnGPU(new) = %v, %v; want placed", placed, err)
	}
	cfg, _ := r.GetConfig("new")
	if cfg.GPUAssign != "1" || cfg.TensorSplit != "0,1,0" || cfg.SplitMode != "layer" || cfg.MainGPU != 1 {
		t.Errorf("config after placing: assign %q split %q mode %q main %d", cfg.GPUAssign, cfg.TensorSplit, cfg.SplitMode, cfg.MainGPU)
	}
	if placed, _ := r.PlaceOnGPU("chosen", "1", 3); placed {
		t.Error("a chosen assignment was replaced")
	}
	if cfg, _ := r.GetConfig("chosen"); cfg.GPUAssign != "all" {
		t.Errorf("chosen assignment is now %q", cfg.GPUAssign)
	}
}
