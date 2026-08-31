package api

import (
	"encoding/csv"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// memoryRuns is the flag-comparison pair with a footprint on the first
// run only: one measured load, one recorded before memory existed.
func memoryRuns() ([]benchmark.BenchmarkRun, jobLookup) {
	runs, jobs := twoBuildRuns()
	runs[0].Memory = &benchmark.MemorySnapshot{
		GPUGiB: 23.0, WeightsGiB: 20.0, KVGiB: 2.0, ComputeGiB: 1.0,
		HostGiB: 1.5, CardDeltaGiB: 24.5, Cards: 4,
	}
	return runs, jobs
}

func TestCSVCellsCarryTheMeasuredFootprint(t *testing.T) {
	runs, jobs := memoryRuns()
	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})

	for col, want := range map[string]string{
		"mem_gpu_gib":     "23",
		"mem_weights_gib": "20",
		"mem_kv_gib":      "2",
		"mem_compute_gib": "1",
		"mem_host_gib":    "1.5",
		"mem_card_gib":    "24.5",
	} {
		idx := columnIndex(t, rows[0], col)
		if rows[1][idx] != want {
			t.Errorf("%s = %q; want %q", col, rows[1][idx], want)
		}
		// The second run measured nothing. Empty, not zero: a zero
		// would read as "this model used no memory".
		if rows[2][idx] != "" {
			t.Errorf("%s on an unmeasured run = %q; want empty", col, rows[2][idx])
		}
	}
}

func TestCSVSummaryCarriesTheMeasuredFootprint(t *testing.T) {
	runs, jobs := memoryRuns()
	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVSummary(cw, runs, jobs)
	})
	idx := columnIndex(t, rows[0], "mem_gpu_gib")
	if rows[1][idx] != "23" {
		t.Errorf("mem_gpu_gib = %q; want 23", rows[1][idx])
	}
	if rows[2][idx] != "" {
		t.Errorf("unmeasured run reports %q", rows[2][idx])
	}
}

// A card figure taken while another model was loading covers both of
// them. The per-instance columns beside it are still sound, so the run
// is exported with the unattributable column blank rather than dropped.
func TestCSVLeavesTheCardColumnBlankForAContendedLoad(t *testing.T) {
	runs, jobs := memoryRuns()
	runs[0].Memory.Contended = true
	runs[0].Memory.CardDeltaGiB = 0

	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})
	if got := rows[1][columnIndex(t, rows[0], "mem_card_gib")]; got != "" {
		t.Errorf("mem_card_gib = %q; want empty for a contended load", got)
	}
	if got := rows[1][columnIndex(t, rows[0], "mem_gpu_gib")]; got != "23" {
		t.Errorf("mem_gpu_gib = %q; the buffer report survives contention", got)
	}
}

func TestJSONExportCarriesTheMeasuredFootprint(t *testing.T) {
	runs, _ := memoryRuns()
	env := ExportEnvelope{Version: exportEnvelopeVersion, Runs: runs}
	b, err := json.Marshal(env)
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)

	if !strings.Contains(out, `"memory":{"gpu_gib":23,"weights_gib":20,"kv_gib":2,"compute_gib":1,"host_gib":1.5,"card_gib":24.5,"cards":4}`) {
		t.Errorf("the footprint is not in the JSON export as expected:\n%s", out)
	}
	// The unmeasured run omits the key entirely rather than exporting a
	// block of zeros.
	if strings.Count(out, `"memory"`) != 1 {
		t.Errorf("an unmeasured run emitted a memory block:\n%s", out)
	}
	if env.Version < 3 {
		t.Error("the envelope version must advertise that runs may carry a footprint")
	}
}
