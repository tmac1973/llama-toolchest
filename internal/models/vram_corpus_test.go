package models

import (
	"math"
	"testing"
)

const gib = 1024 * 1024 * 1024

// corpusPoint is one load measured on hardware: the model as the registry
// would hold it, the config it ran under, and the VRAM the GPU counters
// reported while it was loaded.
//
// This is the evidence the estimate is fitted to. Recorded here rather than
// only in the plan document so a change to the coefficients has to face it.
type corpusPoint struct {
	name     string
	model    Model
	cfg      ModelConfig
	cards    int
	measured float64 // GiB, summed across cards
}

// Sizes are in GiB where written as floats and converted below.
func gibBytes(f float64) int64 { return int64(f * gib) }

func corpus() []corpusPoint {
	// Qwen3.8-Flash-Next: qwen4exp, 48 layers, 12 attending (interval 4),
	// sparse attention with a 128-wide indexer, 26.82 GiB per-layer table.
	fn := func(ctx, ub int) Model {
		return Model{
			SizeBytes: gibBytes(103.69), PLEBytes: gibBytes(26.82),
			TokenEmbdBytes: gibBytes(0.629), NLayers: 48, AttnLayers: 12,
			NEmbd: 2560, NKVHead: 2, KVFullPerTok: 12 * 2 * 512,
			IndexerKeyLength: 128, ContextLength: 262144,
		}
	}
	// Qwen3.8-27B: qwen35, 65 layers, 16 attending, no indexer.
	q27 := Model{
		SizeBytes: gibBytes(29.30), TokenEmbdBytes: gibBytes(1.258),
		NLayers: 65, AttnLayers: 16, NEmbd: 5120, NKVHead: 4,
		KVFullPerTok: 16 * 4 * 512, ContextLength: 262144,
	}
	// Qwen3.6-35B-A3B: qwen35moe, 41 layers, 10 attending.
	q35 := Model{
		SizeBytes: gibBytes(36.41), TokenEmbdBytes: gibBytes(0.503),
		NLayers: 41, AttnLayers: 10, NEmbd: 2048, NKVHead: 2,
		KVFullPerTok: 10 * 2 * 512, ContextLength: 262144,
	}
	// gemma-4-E4B: sliding-window attention, per-layer table, single card.
	gem := Model{
		SizeBytes: gibBytes(4.77), PLEBytes: gibBytes(1.80),
		TokenEmbdBytes: gibBytes(0.664), NLayers: 42, AttnLayers: 42,
		NEmbd: 2560, NKVHead: 2, KVFullPerTok: 14336, KVSWAPerTok: 35840,
		SlidingWindow: 512, ContextLength: 131072,
	}
	c := func(ctx, ub int) ModelConfig { return ModelConfig{ContextSize: ctx, UBatchSize: ub} }
	return []corpusPoint{
		{"Flash-Next ctx32k ub1024", fn(32768, 1024), c(32768, 1024), 4, 84.48},
		{"Flash-Next ctx128k ub1024", fn(131072, 1024), c(131072, 1024), 4, 95.96},
		{"Flash-Next ctx262k ub512", fn(262144, 512), c(262144, 512), 4, 98.96},
		{"27B ctx8k ub512", q27, c(8192, 512), 4, 31.43},
		{"27B ctx32k ub512", q27, c(32768, 512), 4, 33.01},
		{"27B ctx128k ub512", q27, c(131072, 512), 4, 39.38},
		{"27B ctx262k ub512", q27, c(262144, 512), 4, 47.87},
		{"27B ctx32k ub128", q27, c(32768, 128), 4, 32.61},
		{"27B ctx32k ub2048", q27, c(32768, 2048), 4, 34.82},
		{"35B-A3B ctx32k ub512", q35, c(32768, 512), 4, 38.08},
		{"gemma-4-E4B ctx4k ub512", gem, c(4096, 512), 1, 3.42},
	}
}

// The guardrail. An estimate below what the hardware actually used tells
// someone a model fits when it does not, so no point may under-predict —
// however good the average looks.
func TestEstimateNeverUnderPredicts(t *testing.T) {
	for _, p := range corpus() {
		got := VRAMEstimateForConfigOn(&p.model, &p.cfg, p.cards)
		if got < p.measured {
			t.Errorf("%s: estimated %.2f GiB, hardware used %.2f — under by %.2f",
				p.name, got, p.measured, p.measured-got)
		}
	}
}

// Being conservative is only useful if it stays close. A headroom figure
// nobody believes gets ignored, which is the same failure by another route.
func TestEstimateStaysCloseToMeasured(t *testing.T) {
	var sum, worst float64
	var worstName string
	for _, p := range corpus() {
		got := VRAMEstimateForConfigOn(&p.model, &p.cfg, p.cards)
		e := math.Abs(got - p.measured)
		sum += e
		if e > worst {
			worst, worstName = e, p.name
		}
		if e > 3.0 {
			t.Errorf("%s: estimated %.2f against %.2f measured, off by %.2f", p.name, got, p.measured, e)
		}
	}
	mean := sum / float64(len(corpus()))
	if mean > 1.5 {
		t.Errorf("mean error %.2f GiB across the corpus, want under 1.5", mean)
	}
	t.Logf("mean error %.2f GiB, worst %.2f on %s", mean, worst, worstName)
}

// The device count is not cosmetic: scratch and driver overhead are per
// device, so the same config across four cards costs materially more than
// across one. A caller that cannot supply the count gets the single-device
// figure, which is the smaller one — so this must be visible, not silent.
func TestDeviceCountChangesTheEstimate(t *testing.T) {
	p := corpus()[3] // 27B, no indexer
	one := VRAMEstimateForConfigOn(&p.model, &p.cfg, 1)
	four := VRAMEstimateForConfigOn(&p.model, &p.cfg, 4)
	if four <= one {
		t.Fatalf("four cards estimated %.2f, one card %.2f — the count is being ignored", four, one)
	}
	if diff := four - one; diff < 2.0 {
		t.Errorf("four cards cost only %.2f GiB more than one; expected roughly 3x the per-device overhead", diff)
	}
}

// Host-mapped weights are not counted. Both tensors matter: the 27B has no
// per-layer table and still keeps its input embedding off the device.
func TestHostMappedWeightsExcluded(t *testing.T) {
	m := Model{SizeBytes: gibBytes(30.0), PLEBytes: gibBytes(4.0), TokenEmbdBytes: gibBytes(1.0),
		NLayers: 32, AttnLayers: 32, NEmbd: 4096, NKVHead: 8, KVFullPerTok: 32 * 8 * 256}
	cfg := ModelConfig{ContextSize: 4096}
	withBoth := VRAMEstimateForConfigOn(&m, &cfg, 1)

	noEmb := m
	noEmb.TokenEmbdBytes = 0
	if VRAMEstimateForConfigOn(&noEmb, &cfg, 1)-withBoth < 0.9 {
		t.Error("the token embedding is not being excluded from resident weights")
	}
	noPLE := m
	noPLE.PLEBytes = 0
	if VRAMEstimateForConfigOn(&noPLE, &cfg, 1)-withBoth < 3.9 {
		t.Error("the per-layer table is not being excluded from resident weights")
	}
}

// A sparse-attention model pays for ranking the whole cache. Without that
// term Flash-Next is under-predicted by tens of GiB at long context.
func TestIndexerTermOnlyAppliesToRankingModels(t *testing.T) {
	p := corpus()[2] // Flash-Next ctx262k
	with := VRAMEstimateForConfigOn(&p.model, &p.cfg, p.cards)

	plain := p.model
	plain.IndexerKeyLength = 0
	without := VRAMEstimateForConfigOn(&plain, &p.cfg, p.cards)

	if with-without < 10 {
		t.Errorf("indexer term worth only %.2f GiB at 262144 context; measured compute alone was 24 GiB", with-without)
	}
	if without >= p.measured {
		t.Error("the estimate is adequate without the indexer term; the fixture no longer shows why it exists")
	}
}
