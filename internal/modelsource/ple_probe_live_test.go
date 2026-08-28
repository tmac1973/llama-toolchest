package modelsource

import (
	"context"
	"net/http"
	"os"
	"testing"
	"time"
)

// The probe's whole justification is that the header of a large split
// GGUF is cheap to reach. This checks that against the real thing rather
// than a fixture, so a change in how models are split shows up here.
// Opt-in: MODELSCOPE_LIVE_TEST=1.
func TestLiveProbeQwen38FlashNext(t *testing.T) {
	if os.Getenv("MODELSCOPE_LIVE_TEST") != "1" {
		t.Skip("set MODELSCOPE_LIVE_TEST=1 to run against live model hosts")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// HuggingFace rather than ModelScope: the same repository, and the
	// probe is source-agnostic, but ModelScope rate-limits an address
	// that has been probing it repeatedly, which is exactly what
	// developing this feature involves.
	const base = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-"
	shards := []ShardURL{
		{URL: base + "00001-of-00003.gguf", Size: 10946624},
		{URL: base + "00002-of-00003.gguf", Size: 49990818368},
		{URL: base + "00003-of-00003.gguf", Size: 22544696352},
	}
	var total int64
	for _, s := range shards {
		total += s.Size
	}

	start := time.Now()
	got := ProbePLE(ctx, &http.Client{Timeout: 60 * time.Second}, "", shards, total)
	elapsed := time.Since(start)

	if !got.Probed {
		t.Fatal("probe did not complete")
	}
	if got.StreamedBytes == 0 {
		t.Fatal("no per-layer embedding table found in Qwen3.8-Flash-Next, which has one")
	}
	streamedGiB := float64(got.StreamedBytes) / (1 << 30)
	totalGiB := float64(total) / (1 << 30)
	if streamedGiB < 20 || streamedGiB > 35 {
		t.Errorf("streamed = %.2f GiB, expected roughly 27 (about 40%% of the download)", streamedGiB)
	}
	t.Logf("total %.1f GiB, streamed %.2f GiB (%.0f%%), uncorrected estimate %.1f GiB, corrected %.1f GiB, probe took %s",
		totalGiB, streamedGiB, streamedGiB/totalGiB*100,
		totalGiB*1.1, (totalGiB-streamedGiB)*1.1, elapsed.Round(time.Millisecond))
}
