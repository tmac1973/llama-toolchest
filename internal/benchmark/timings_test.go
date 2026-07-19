package benchmark

import (
	"math"
	"testing"
)

// sample builds a TimingSample from tokens and the seconds they took.
func sample(model string, promptTok int, promptSecs float64, genTok int, genSecs float64) TimingSample {
	return TimingSample{
		ModelID:         model,
		PromptTokens:    promptTok,
		PromptTokPerSec: float64(promptTok) / promptSecs,
		GenTokens:       genTok,
		GenTokPerSec:    float64(genTok) / genSecs,
	}
}

// The exact scenario that made a faster server look slower: a window of
// mostly-small cache-hit continuations alongside a few large prompts.
// Averaging per-request rates gave a 72-token prefill the same weight as
// a 2107-token one, so the headline figure tracked the traffic mix
// rather than the hardware.
func TestTimingSummaryIsTokenWeighted(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{}}
	// 11 small requests at ~190 t/s, 1 large at ~1220 t/s.
	for i := 0; i < 11; i++ {
		s.timings["m"] = append(s.timings["m"], sample("m", 150, 150.0/190, 20, 20.0/30))
	}
	s.timings["m"] = append(s.timings["m"], sample("m", 2107, 2107.0/1220, 100, 100.0/30))

	got := s.TimingSummary()
	if len(got) != 1 {
		t.Fatalf("got %d summaries, want 1", len(got))
	}

	// Unweighted mean of rates would be ~276. Token-weighted is
	// (11*150 + 2107) / (11*150/190 + 2107/1220) = ~480.
	unweighted := (11*190.0 + 1220.0) / 12
	if math.Abs(got[0].AvgPromptTokPerSec-unweighted) < 20 {
		t.Errorf("PP rate %.0f looks like the unweighted mean (%.0f); it must be token-weighted",
			got[0].AvgPromptTokPerSec, unweighted)
	}

	wantTok := 11*150.0 + 2107
	wantSecs := 11*(150.0/190) + 2107.0/1220
	want := wantTok / wantSecs
	if math.Abs(got[0].AvgPromptTokPerSec-want) > 1 {
		t.Errorf("PP rate = %.1f, want %.1f (total tokens / total time)", got[0].AvgPromptTokPerSec, want)
	}
}

// Per-size rows are what make two periods comparable when the mix moved.
func TestTimingSummaryBucketsByPromptSize(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{"m": {
		sample("m", 100, 1, 10, 1),   // <512
		sample("m", 1000, 1, 10, 1),  // 512–2k
		sample("m", 4000, 1, 10, 1),  // 2k–8k
		sample("m", 20000, 1, 10, 1), // 8k+
	}}}

	got := s.TimingSummary()[0]
	if len(got.Buckets) != 4 {
		t.Fatalf("got %d buckets, want one per populated range: %+v", len(got.Buckets), got.Buckets)
	}
	for _, b := range got.Buckets {
		if b.Count != 1 {
			t.Errorf("bucket %s has %d samples, want 1", b.Label, b.Count)
		}
	}
	// Each bucket's rate is its own, not the overall figure.
	if got.Buckets[0].AvgPromptTokPerSec != 100 || got.Buckets[3].AvgPromptTokPerSec != 20000 {
		t.Errorf("bucket rates wrong: %.0f and %.0f", got.Buckets[0].AvgPromptTokPerSec, got.Buckets[3].AvgPromptTokPerSec)
	}
}

// The prompt-length range stops a headline number being read as a
// hardware measurement when it only covers tiny requests.
func TestTimingSummaryReportsPromptRange(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{"m": {
		sample("m", 72, 0.7, 10, 1),
		sample("m", 2210, 4.9, 10, 1),
	}}}
	got := s.TimingSummary()[0]
	if got.MinPromptTokens != 72 || got.MaxPromptTokens != 2210 {
		t.Errorf("range = %d–%d, want 72–2210", got.MinPromptTokens, got.MaxPromptTokens)
	}
}

// A sample with no prompt tokens (pure continuation) must not divide by
// zero or drag the rate to zero.
func TestTimingSummaryIgnoresEmptySamples(t *testing.T) {
	s := &Store{timings: map[string][]TimingSample{"m": {
		sample("m", 1000, 1, 10, 1),
		{ModelID: "m", PromptTokens: 0, PromptTokPerSec: 0, GenTokens: 10, GenTokPerSec: 30},
	}}}
	got := s.TimingSummary()[0]
	if got.AvgPromptTokPerSec != 1000 {
		t.Errorf("PP rate = %.1f, want 1000 — the empty sample should not contribute", got.AvgPromptTokPerSec)
	}
	if got.Count != 2 {
		t.Errorf("Count = %d, want both samples counted", got.Count)
	}
}

// Per-size reporting is the only form comparable to llama-bench, which
// publishes one figure per fixed prompt length (pp512, pp2048) and never
// averages across sizes — prompt throughput rises steeply with length,
// so a mixed average matches nothing published anywhere.
func TestComputeSummaryReportsPerSize(t *testing.T) {
	results := []BenchmarkResult{
		{PromptTokens: 512, GenTokens: 128, PromptTokPerSec: 1000, GenTokPerSec: 30},
		{PromptTokens: 512, GenTokens: 128, PromptTokPerSec: 1100, GenTokPerSec: 32},
		{PromptTokens: 2048, GenTokens: 128, PromptTokPerSec: 2000, GenTokPerSec: 28},
		{PromptTokens: 2048, GenTokens: 128, PromptTokPerSec: 2200, GenTokPerSec: 30},
	}
	got := ComputeSummary(results)

	if len(got.PerSize) != 2 {
		t.Fatalf("got %d size groups, want 2: %+v", len(got.PerSize), got.PerSize)
	}
	// Sorted ascending by prompt length.
	if got.PerSize[0].PromptTokens != 512 || got.PerSize[1].PromptTokens != 2048 {
		t.Errorf("sizes = %d, %d; want 512 then 2048",
			got.PerSize[0].PromptTokens, got.PerSize[1].PromptTokens)
	}
	if got.PerSize[0].PPMean != 1050 || got.PerSize[1].PPMean != 2100 {
		t.Errorf("means = %.0f, %.0f; want 1050 and 2100",
			got.PerSize[0].PPMean, got.PerSize[1].PPMean)
	}
	// Standard deviation is what tells a real difference from noise.
	if got.PerSize[0].PPStd != 50 {
		t.Errorf("stddev = %.1f, want 50", got.PerSize[0].PPStd)
	}
	if got.PerSize[0].Label() != "pp512" {
		t.Errorf("label = %q, want pp512", got.PerSize[0].Label())
	}
}

// The mixed average is retained for stored data but must not be mistaken
// for a per-size figure.
func TestComputeSummaryKeepsMixedAverage(t *testing.T) {
	results := []BenchmarkResult{
		{PromptTokens: 512, PromptTokPerSec: 1000, GenTokPerSec: 30},
		{PromptTokens: 8192, PromptTokPerSec: 3000, GenTokPerSec: 30},
	}
	got := ComputeSummary(results)
	if got.AvgPromptTokPerSec != 2000 {
		t.Errorf("mixed average = %.0f, want 2000", got.AvgPromptTokPerSec)
	}
	if got.PerSize[0].PPMean == got.AvgPromptTokPerSec {
		t.Error("per-size figures must not equal the mixed average")
	}
}

func TestComputeSummarySingleSizeHasZeroStdOnOneRep(t *testing.T) {
	got := ComputeSummary([]BenchmarkResult{
		{PromptTokens: 512, PromptTokPerSec: 1000, GenTokPerSec: 30},
	})
	if len(got.PerSize) != 1 || got.PerSize[0].PPStd != 0 {
		t.Errorf("one repetition should report zero spread, got %+v", got.PerSize)
	}
}
