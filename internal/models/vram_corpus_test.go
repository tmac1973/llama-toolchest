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
	// reported is the same load as llama.cpp itemised it while
	// allocating, GPU side only. A total can hide two errors that cancel
	// — the estimate this corpus replaced did exactly that — so where the
	// split is known it is recorded, and TestEstimateTermsAgainstTheBufferReport
	// checks the terms one at a time.
	//
	// Nil for points measured before the split was captured. New points
	// should carry it: /api/models/{id}/vram-corpus prints a filled-in
	// row for whatever the router last loaded.
	reported *reportedTerms
}

// reportedTerms is one load's buffer report, GiB on the accelerators.
type reportedTerms struct {
	weights   float64
	kv        float64
	recurrent float64
	compute   float64 // compute and output buffers
}

// total is what llama.cpp accounts for. Always less than the card
// counters: context and allocator overhead are not in the report.
func (r reportedTerms) total() float64 {
	return r.weights + r.kv + r.recurrent + r.compute
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
	// The 27B's weights and recurrent state are the same on every row —
	// neither depends on context or micro-batch, which is itself part of
	// what the sweep established.
	q27terms := func(kv, compute float64) *reportedTerms {
		return &reportedTerms{weights: 27.51, kv: kv, recurrent: 0.60, compute: compute}
	}
	return []corpusPoint{
		{"Flash-Next ctx32k ub1024", fn(32768, 1024), c(32768, 1024), 4, 84.48, nil},
		{"Flash-Next ctx128k ub1024", fn(131072, 1024), c(131072, 1024), 4, 95.96, nil},
		{"Flash-Next ctx262k ub512", fn(262144, 512), c(262144, 512), 4, 98.96, nil},
		// The one sparse-attention load that was decomposed: its saved
		// config, quantized KV cache and all. See "The decomposition,
		// measured" in plan/ple-vram-findings.md.
		{"Flash-Next ctx262k ub1024 kv-q8_0", fn(262144, 1024),
			ModelConfig{ContextSize: 262144, UBatchSize: 1024, KVCacheQuant: "q8_0"}, 4, 107.35,
			&reportedTerms{weights: 76.23, kv: 4.38, recurrent: 0.44, compute: 24.40}},
		{"27B ctx8k ub512", q27, c(8192, 512), 4, 31.43, q27terms(0.48, 0.52)},
		{"27B ctx32k ub512", q27, c(32768, 512), 4, 33.01, q27terms(2.00, 0.60)},
		{"27B ctx128k ub512", q27, c(131072, 512), 4, 39.38, q27terms(8.00, 0.96)},
		{"27B ctx262k ub512", q27, c(262144, 512), 4, 47.87, q27terms(16.00, 1.48)},
		{"27B ctx32k ub128", q27, c(32768, 128), 4, 32.61, q27terms(2.00, 0.20)},
		{"27B ctx32k ub2048", q27, c(32768, 2048), 4, 34.82, q27terms(2.00, 2.40)},
		{"35B-A3B ctx32k ub512", q35, c(32768, 512), 4, 38.08, nil},
		{"gemma-4-E4B ctx4k ub512", gem, c(4096, 512), 1, 3.42, nil},
	}
}

// point looks a corpus point up by name. Positional access breaks
// silently when a point is inserted, and inserting points is the whole
// intent of this file.
func point(t *testing.T, name string) corpusPoint {
	t.Helper()
	for _, p := range corpus() {
		if p.name == name {
			return p
		}
	}
	t.Fatalf("no corpus point named %q", name)
	return corpusPoint{}
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
	p := point(t, "27B ctx262k ub512") // no indexer
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
	p := point(t, "Flash-Next ctx262k ub512")
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

// The estimate has to be right for the right reasons. A total alone
// cannot tell a good model from two mistakes that cancel: the estimate
// this corpus replaced was within 7 GiB on Qwen3.8-Flash-Next while
// over-counting weights by 27 and under-counting everything else by 20.
//
// So each term is checked against what llama.cpp said it allocated, in
// the same direction as the total: at or above measured, and close.
func TestEstimateTermsAgainstTheBufferReport(t *testing.T) {
	const closeEnough = 1.0 // GiB, per term

	checked := 0
	for _, p := range corpus() {
		if p.reported == nil {
			continue
		}
		checked++
		b := VRAMBreakdownForConfigOn(&p.model, &p.cfg, p.cards)

		weights := b.Weights + b.Aux
		if weights < p.reported.weights {
			t.Errorf("%s: weights estimated %.2f, measured %.2f — under by %.2f",
				p.name, weights, p.reported.weights, p.reported.weights-weights)
		}
		if weights-p.reported.weights > closeEnough {
			t.Errorf("%s: weights over by %.2f GiB", p.name, weights-p.reported.weights)
		}

		// The KV term answers to the attention cache only; recurrent
		// state is the term below.
		kv := b.KVCache + b.IndexerCache
		if kv < p.reported.kv {
			t.Errorf("%s: KV cache estimated %.2f, measured %.2f — under by %.2f",
				p.name, kv, p.reported.kv, p.reported.kv-kv)
		}
		if kv-p.reported.kv > closeEnough {
			t.Errorf("%s: KV cache over by %.2f GiB", p.name, kv-p.reported.kv)
		}

		compute := b.Compute + b.IndexerScratch
		if compute < p.reported.compute {
			t.Errorf("%s: compute buffers estimated %.2f, measured %.2f — under by %.2f",
				p.name, compute, p.reported.compute, p.reported.compute-compute)
		}
		if compute-p.reported.compute > closeEnough {
			t.Errorf("%s: compute buffers over by %.2f GiB", p.name, compute-p.reported.compute)
		}

		if diff := math.Abs(b.Reported() - p.reported.total()); diff > closeEnough {
			t.Errorf("%s: llama.cpp accounts for %.2f GiB, the modelled terms for %.2f — off by %.2f",
				p.name, p.reported.total(), b.Reported(), diff)
		}
	}
	if checked == 0 {
		t.Fatal("no corpus point carries a buffer report; this test proves nothing")
	}
	t.Logf("checked %d of %d corpus points term by term", checked, len(corpus()))
}

// Recurrent state is real and is not modelled: every hybrid in the corpus
// allocates some, and no term accounts for it. What makes that safe is
// the per-device overhead, which has to cover both the state and the gap
// between llama.cpp's accounting and the card counters.
//
// If a recurrent term is ever added, this test should be deleted rather
// than adjusted — it exists to say the omission is deliberate and paid
// for, not that it is correct.
func TestUnmodelledRecurrentStateIsCoveredByOverhead(t *testing.T) {
	for _, p := range corpus() {
		if p.reported == nil || p.reported.recurrent == 0 {
			continue
		}
		b := VRAMBreakdownForConfigOn(&p.model, &p.cfg, p.cards)

		// Everything the estimate does not itemise: the recurrent state
		// it has no term for, plus what llama.cpp itself never reports.
		unaccounted := p.reported.recurrent + (p.measured - p.reported.total())
		if b.Overhead < unaccounted {
			t.Errorf("%s: overhead allows %.2f GiB but %.2f is unaccounted for (%.2f recurrent state, %.2f above the buffer report)",
				p.name, b.Overhead, unaccounted, p.reported.recurrent, p.measured-p.reported.total())
		}
	}
}

// The buffer report is not the whole story: context and allocator
// overhead never appear in it. Anything reading the report as the
// footprint — the Available Models tooltip, a corpus row — is reading a
// figure below what the card will show, and the estimate has to keep
// covering that difference.
func TestCardCountersExceedTheBufferReport(t *testing.T) {
	for _, p := range corpus() {
		if p.reported == nil {
			continue
		}
		remainder := p.measured - p.reported.total()
		if remainder <= 0 {
			t.Errorf("%s: the buffer report (%.2f) is not below the card counters (%.2f) — one of the two figures is wrong",
				p.name, p.reported.total(), p.measured)
		}
		if remainder > 3.0 {
			t.Errorf("%s: %.2f GiB unreported; the remainder was a near-constant 2.3 across the sweep it was measured on", p.name, remainder)
		}
	}
}
