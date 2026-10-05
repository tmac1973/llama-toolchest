package benchmark

import "testing"

// summarizeBenchy fills the run's list-view summary from llama-benchy's
// results. It reads the concurrency-1 result when there is one, because
// that is the figure comparable to the internal runner; takes min and max
// generation speed from the per-run values around the mean; and uses
// end-to-end TTFT, falling back to time to first response only when
// llama-benchy did not report it. With no results there is no summary.
func TestSummarizeBenchy(t *testing.T) {
	m := func(mean float64, values ...float64) *LlamaBenchyMetric {
		return &LlamaBenchyMetric{Mean: mean, Values: values}
	}
	cases := []struct {
		name    string
		results []LlamaBenchyResult
		want    *BenchmarkSummary
	}{
		{name: "nil results", results: nil, want: nil},
		{name: "empty results", results: []LlamaBenchyResult{}, want: nil},
		{
			name: "concurrency 1 preferred over an earlier result",
			results: []LlamaBenchyResult{
				{Concurrency: 4, PPThroughput: m(4000), TGThroughput: m(200), E2ETTFT: m(900)},
				{Concurrency: 1, PPThroughput: m(1500), TGThroughput: m(60), E2ETTFT: m(120)},
				{Concurrency: 1, PPThroughput: m(1), TGThroughput: m(1), E2ETTFT: m(1)},
			},
			want: &BenchmarkSummary{AvgPromptTokPerSec: 1500, AvgGenTokPerSec: 60, MinGenTokPerSec: 60, MaxGenTokPerSec: 60, AvgTTFTMs: 120},
		},
		{
			name: "first result used when none has concurrency 1",
			results: []LlamaBenchyResult{
				{Concurrency: 2, PPThroughput: m(2500), TGThroughput: m(110)},
				{Concurrency: 8, PPThroughput: m(6000), TGThroughput: m(300)},
			},
			want: &BenchmarkSummary{AvgPromptTokPerSec: 2500, AvgGenTokPerSec: 110, MinGenTokPerSec: 110, MaxGenTokPerSec: 110},
		},
		{
			name: "min and max come from the per-run values",
			results: []LlamaBenchyResult{
				{Concurrency: 1, TGThroughput: m(60, 58.5, 61.25, 60.25, 59)},
			},
			want: &BenchmarkSummary{AvgGenTokPerSec: 60, MinGenTokPerSec: 58.5, MaxGenTokPerSec: 61.25},
		},
		{
			name: "E2E TTFT wins over TTFR",
			results: []LlamaBenchyResult{
				{Concurrency: 1, TTFR: m(80), E2ETTFT: m(95)},
			},
			want: &BenchmarkSummary{AvgTTFTMs: 95},
		},
		{
			name: "TTFR used when E2E TTFT is missing",
			results: []LlamaBenchyResult{
				{Concurrency: 1, TTFR: m(80)},
			},
			want: &BenchmarkSummary{AvgTTFTMs: 80},
		},
		{
			// Every metric is nullable in llama-benchy's schema; a result
			// with none of them gives zeros, not a crash.
			name:    "missing metrics leave zeros",
			results: []LlamaBenchyResult{{Concurrency: 1}},
			want:    &BenchmarkSummary{},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := summarizeBenchy(c.results)
			if c.want == nil {
				if got != nil {
					t.Fatalf("summarizeBenchy = %+v, want nil", got)
				}
				return
			}
			if got == nil {
				t.Fatalf("summarizeBenchy = nil, want %+v", c.want)
			}
			if got.AvgPromptTokPerSec != c.want.AvgPromptTokPerSec ||
				got.AvgGenTokPerSec != c.want.AvgGenTokPerSec ||
				got.MinGenTokPerSec != c.want.MinGenTokPerSec ||
				got.MaxGenTokPerSec != c.want.MaxGenTokPerSec ||
				got.AvgTTFTMs != c.want.AvgTTFTMs {
				t.Errorf("summarizeBenchy = %+v\nwant %+v", *got, *c.want)
			}
		})
	}
}
