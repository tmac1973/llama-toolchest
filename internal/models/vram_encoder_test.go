package models

import "testing"

// The two encoders measured on asmodean (R9700, ROCm, 2026-11-03) with
// llama.cpp's buffer report, at the launch config toolchest gives them.
// "Real" is the sum of the reported buffers on upstream b11887 plus the
// 0.35 GiB the card showed above that sum: the estimate must cover it,
// and stay within 15% above it.
func TestVRAMEstimateEncoders(t *testing.T) {
	reranker := &Model{
		ID: "ggml-org--Qwen3-Reranker-0.6B-Q8_0-GGUF--qwen3-reranker-0.6b-q8_0", ModelID: "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
		Arch: "qwen3", SizeBytes: 639153184, TokenEmbdBytes: 165015872, OutputTied: true,
		NLayers: 28, AttnLayers: 28, NEmbd: 1024, NHead: 16, NKVHead: 8, ContextLength: 40960,
		KVFullPerTok: 57344, PoolingType: PoolingRank, HasClassifierHead: true,
	}
	embedder := &Model{
		ID: "ggml-org--embeddinggemma-2-GGUF--embeddinggemma-2-Q8_0", ModelID: "ggml-org/embeddinggemma-2-GGUF",
		Arch: "gemma-embedding2", SizeBytes: 309855456, TokenEmbdBytes: 142606336,
		NLayers: 24, AttnLayers: 24, NEmbd: 512, NHead: 4, NKVHead: 2, ContextLength: 262144,
		KVFullPerTok: 4096, KVSWAPerTok: 20480, SlidingWindow: 1024,
		PoolingType: PoolingMean, Embedding: true, NonCausal: true,
	}
	const overhead = 0.35
	cases := []struct {
		name string
		m    *Model
		cfg  *ModelConfig
		// weights, KV and compute buffers in MiB, as llama.cpp reported
		weights, kv, compute float64
	}{
		// The config asmodean runs: context 32768, so the micro-batch too.
		{"reranker 32K", reranker, &ModelConfig{ContextSize: 32768, GPULayers: 999}, 603.87, 3584, 3713.00},
		{"reranker 8K", reranker, &ModelConfig{ContextSize: 8192, GPULayers: 999}, 603.87, 896, 544.25},
		// The default: EmbeddingDefaultContext.
		{"embedder default", embedder, &ModelConfig{}, 144.44, 0, 1088.28},
		{"embedder 16K", embedder, &ModelConfig{ContextSize: 16384}, 144.44, 0, 2688.56},
	}
	for _, c := range cases {
		real := (c.weights+c.kv+c.compute)/1024 + overhead
		b := vramBreakdownWith(&rocmCoefficients, c.m, c.cfg, 1)
		got := b.Total()
		if got < real || got > real*1.15 {
			t.Errorf("%s: estimate %.2f GiB, real %.2f GiB (%+.0f%%)\n  %+v", c.name, got, real, (got/real-1)*100, b)
		}
		if c.kv == 0 && b.KVCache != 0 {
			t.Errorf("%s: KV cache %.2f GiB for a non-causal model", c.name, b.KVCache)
		}
	}
}

// The encoder rule follows the width: a wider Qwen3 embedding model gets
// proportionally more scratch per token, and an architecture without a
// measurement the larger figure.
func TestEncoderComputeScales(t *testing.T) {
	co := &rocmCoefficients
	small := encoderComputeGiB(co, &Model{Arch: "qwen3", NEmbd: 1024}, 512, 32768)
	wide := encoderComputeGiB(co, &Model{Arch: "qwen3", NEmbd: 4096}, 512, 32768)
	if wide <= small {
		t.Errorf("Qwen3 at n_embd 4096 = %.3f GiB, not above n_embd 1024 = %.3f", wide, small)
	}
	known := encoderComputeGiB(co, &Model{Arch: "qwen3", NEmbd: 1024}, 8192, 8192)
	unknown := encoderComputeGiB(co, &Model{Arch: "bert", NEmbd: 1024}, 8192, 8192)
	if unknown <= known {
		t.Errorf("unknown architecture %.3f GiB, not above qwen3 %.3f", unknown, known)
	}
	// Two masks with sliding-window layers.
	one := encoderComputeGiB(co, &Model{Arch: "bert", NEmbd: 768}, 8192, 8192)
	two := encoderComputeGiB(co, &Model{Arch: "bert", NEmbd: 768, SlidingWindow: 512}, 8192, 8192)
	if d := two - one; d < 0.124 || d > 0.126 {
		t.Errorf("second mask adds %.3f GiB, want 0.125 (8192² f16)", d)
	}
}

// A chat model's estimate does not change: the encoder rules apply only
// to embedding and reranking models.
func TestVRAMEstimateChatUnchanged(t *testing.T) {
	chat := &Model{Arch: "qwen3", SizeBytes: 639153184, TokenEmbdBytes: 165015872, OutputTied: true,
		NLayers: 28, AttnLayers: 28, NEmbd: 1024, NHead: 16, NKVHead: 8, ContextLength: 40960, KVFullPerTok: 57344}
	cfg := &ModelConfig{ContextSize: 32768}
	b := vramBreakdownWith(&rocmCoefficients, chat, cfg, 1)
	if want := rocmCoefficients.single.at(cfg.EffectiveUBatchSize(), 32768); b.Compute != want {
		t.Errorf("chat compute %.3f, want the chat fit %.3f", b.Compute, want)
	}
	if b.KVCache == 0 {
		t.Error("chat model lost its KV cache")
	}
}
