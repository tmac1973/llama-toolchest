package api

import (
	"go/parser"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/memreport"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// corpusTestLoad feeds one load through a collector and hands back what
// the exporter works from.
func corpusTestLoad(t *testing.T, note loadNote) (memreport.Measurement, loadNote) {
	t.Helper()
	c := memreport.NewCollector()
	for _, line := range strings.Split(oneLoad, "\n") {
		if ev := c.Add(line); ev.Kind == memreport.EventSpawned {
			c.Annotate(ev.Port, note)
		}
	}
	meas, ok := c.Latest("m1")
	if !ok {
		t.Fatal("no measurement")
	}
	return meas, note
}

func TestCorpusRowIsValidGo(t *testing.T) {
	cfg := &models.ModelConfig{ContextSize: 8192, UBatchSize: 512, KVCacheQuant: "q8_0"}
	meas, note := corpusTestLoad(t, loadNote{
		modelID: "m1", cards: 1, cfg: cfg,
		baselineGiB: 1.5, loadedGiB: 26.0,
	})

	row := corpusRow(memTestModel(), meas, note, "b10679-rocm")

	// The whole point is pasting it into a Go table, so it has to parse
	// as one. Comments and all.
	if _, err := parser.ParseExpr("[]corpusPoint{\n" + row + "}"); err != nil {
		t.Fatalf("exported row is not valid Go: %v\nrow:\n%s", err, row)
	}
}

func TestCorpusRowCarriesTheMeasurementAndItsTerms(t *testing.T) {
	cfg := &models.ModelConfig{ContextSize: 8192, UBatchSize: 512}
	meas, note := corpusTestLoad(t, loadNote{
		modelID: "m1", cards: 1, cfg: cfg,
		baselineGiB: 1.5, loadedGiB: 26.0,
	})

	row := corpusRow(memTestModel(), meas, note, "b10679-rocm")

	for _, want := range []string{
		"build b10679-rocm",
		// 26.0 - 1.5 card counters, against 23.0 llama.cpp itemised.
		"Card counters rose 24.50 GiB; llama.cpp itemised 23.00, leaving 1.50 unreported.",
		"ctx8k ub512",
		"ContextSize: 8192, UBatchSize: 512",
		"&reportedTerms{weights: 20.00, kv: 2.00, recurrent: 0.00, compute: 1.00}",
		", 1, 24.50,",
		"A further 1.00 GiB went to system memory",
	} {
		if !strings.Contains(row, want) {
			t.Errorf("row missing %q; got:\n%s", want, row)
		}
	}
}

// A row measured while something else was loading is worse than no row:
// it looks like evidence and is not.
func TestCorpusRowRefusesToPassOffAContendedLoad(t *testing.T) {
	cfg := &models.ModelConfig{ContextSize: 8192, UBatchSize: 512}
	meas, note := corpusTestLoad(t, loadNote{
		modelID: "m1", cards: 1, cfg: cfg,
		baselineGiB: 1.5, loadedGiB: 40.0, contended: true,
	})

	row := corpusRow(memTestModel(), meas, note, "")
	if !strings.Contains(row, "NOT USABLE AS MEASURED") {
		t.Errorf("a contended load must say so; got:\n%s", row)
	}
	if !strings.Contains(row, "buffer report is still this model's own") {
		t.Errorf("the per-instance figures survive contention and the row should say which half is sound; got:\n%s", row)
	}
}

func TestCorpusRowWithoutCardCounters(t *testing.T) {
	cfg := &models.ModelConfig{ContextSize: 8192, UBatchSize: 512}
	meas, note := corpusTestLoad(t, loadNote{modelID: "m1", cards: 1, cfg: cfg})

	row := corpusRow(memTestModel(), meas, note, "")
	if !strings.Contains(row, "filled in by hand") {
		t.Errorf("a row with no card figure must say so rather than claiming 0 GiB; got:\n%s", row)
	}
	if !strings.Contains(row, ", 1, 0.00,") {
		t.Errorf("the measured column should be an obvious zero; got:\n%s", row)
	}
}

// A tensor-parallel load reports per card, and a corpus row states
// totals. Getting this wrong quarters a four-card model.
func TestCorpusRowScalesAggregatedFigures(t *testing.T) {
	const split = `0.00.100.000 I srv          load: spawning server instance with name=m1 on port 46231
[46231] 0.14.275.753 I load_tensors:       Meta() model buffer size = 5120.00 MiB
[46231] 0.16.162.940 I llama_kv_cache:     Meta() KV buffer size = 1024.00 MiB`

	c := memreport.NewCollector()
	note := loadNote{modelID: "m1", cards: 4, cfg: &models.ModelConfig{ContextSize: 8192, UBatchSize: 512},
		baselineGiB: 2.0, loadedGiB: 28.0}
	for _, line := range strings.Split(split, "\n") {
		if ev := c.Add(line); ev.Kind == memreport.EventSpawned {
			c.Annotate(ev.Port, note)
		}
	}
	meas, _ := c.Latest("m1")

	row := corpusRow(memTestModel(), meas, note, "")
	if !strings.Contains(row, "weights: 20.00") {
		t.Errorf("4 x 5120 MiB = 20 GiB of weights; got:\n%s", row)
	}
	if !strings.Contains(row, ", 4, 26.00,") {
		t.Errorf("row should state 4 cards and the 26 GiB the counters rose; got:\n%s", row)
	}
}

func TestUsedGiBSumsEveryCard(t *testing.T) {
	got := usedGiB(monitor.Metrics{GPU: []monitor.GPUInfo{
		{VRAMUsedMB: 1024}, {VRAMUsedMB: 2048}, {VRAMUsedMB: 512},
	}})
	if got != 3.5 {
		t.Errorf("used = %v GiB; want 3.5", got)
	}
}

// A load that finishes while the monitor is not running must not block
// the goroutine waiting on it forever.
func TestGPUSampleGivesUpWhenTheMonitorIsQuiet(t *testing.T) {
	s := memTestServer(t, "4")
	start := time.Now()
	if _, ok := s.gpuUsedAfterFreshSample(50 * time.Millisecond); ok {
		t.Error("a sample was reported by a monitor that never polled")
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("waited %v for a 50ms timeout", elapsed)
	}
}
