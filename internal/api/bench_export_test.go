package api

import (
	"bytes"
	"encoding/csv"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// twoBuildRuns models the flag-comparison workflow: two runs of the same
// model at the same git ref, differing only in CMake flags. Before
// cmake_flags was exported these rows were indistinguishable in CSV.
func twoBuildRuns() ([]benchmark.BenchmarkRun, jobLookup) {
	job := &benchmark.BenchmarkJob{ID: "job-1", Name: "fattn compare"}
	common := map[string]string{
		"GGML_HIP":         "ON",
		"CMAKE_BUILD_TYPE": "Release",
	}
	withFattn := map[string]string{}
	for k, v := range common {
		withFattn[k] = v
	}
	withFattn["GGML_HIP_ROCWMMA_FATTN"] = "ON"

	mk := func(id string, flags map[string]string, pp float64) benchmark.BenchmarkRun {
		return benchmark.BenchmarkRun{
			ID:        id,
			JobID:     "job-1",
			CreatedAt: time.Unix(0, 0).UTC(),
			Status:    "completed",
			ModelID:   "m",
			ModelName: "Qwen3.6-27B-MTP",
			Preset:    "internal-long-ctx",
			Build: benchmark.BuildSnapshot{
				ID: id, Profile: "rocm", GitRef: "b10068", CMakeFlags: flags,
			},
			Results: []benchmark.BenchmarkResult{{
				PromptTokens: 25125, GenTokens: 512, Repetition: 1,
				PromptTokPerSec: pp,
			}},
			Summary: &benchmark.BenchmarkSummary{AvgPromptTokPerSec: pp},
		}
	}

	return []benchmark.BenchmarkRun{
			mk("b10068-rocm", withFattn, 2143.5),
			mk("b10068-rocm-plain", common, 1726.4),
		}, jobLookup{"job-1": job}
}

func parseCSV(t *testing.T, write func(*csv.Writer) error) [][]string {
	t.Helper()
	var buf bytes.Buffer
	cw := csv.NewWriter(&buf)
	if err := write(cw); err != nil {
		t.Fatalf("write: %v", err)
	}
	cw.Flush()
	rows, err := csv.NewReader(&buf).ReadAll()
	if err != nil {
		// encoding/csv rejects ragged rows, so this fires if any data
		// row's width drifts from the header's.
		t.Fatalf("parse: %v", err)
	}
	if len(rows) < 2 {
		t.Fatalf("expected header + data rows, got %d", len(rows))
	}
	return rows
}

func columnIndex(t *testing.T, header []string, name string) int {
	t.Helper()
	for i, h := range header {
		if h == name {
			return i
		}
	}
	t.Fatalf("column %q not found in header %v", name, header)
	return -1
}

func TestCSVCellsIncludesCMakeFlags(t *testing.T) {
	runs, jobs := twoBuildRuns()
	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})

	idx := columnIndex(t, rows[0], "cmake_flags")
	got := []string{rows[1][idx], rows[2][idx]}

	want := []string{
		"-DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON",
		"-DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON",
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d cmake_flags = %q, want %q", i+1, got[i], want[i])
		}
	}
	if got[0] == got[1] {
		t.Error("the two builds must be distinguishable in CSV")
	}
}

func TestCSVSummaryIncludesCMakeFlags(t *testing.T) {
	runs, jobs := twoBuildRuns()
	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVSummary(cw, runs, jobs)
	})

	idx := columnIndex(t, rows[0], "cmake_flags")
	if rows[1][idx] == rows[2][idx] {
		t.Errorf("both summary rows report cmake_flags %q", rows[1][idx])
	}
	if rows[1][idx] == "" {
		t.Error("cmake_flags empty for a build that has flags")
	}
}

// A run that failed before producing results still emits one row; its
// width must match the header like every other row.
func TestCSVCellsEmptyResultsRowWidth(t *testing.T) {
	runs, jobs := twoBuildRuns()
	runs[0].Results = nil
	runs[0].Summary = nil

	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})
	for i, row := range rows {
		if len(row) != len(rows[0]) {
			t.Errorf("row %d has %d fields, header has %d", i, len(row), len(rows[0]))
		}
	}
}

func TestFormatCMakeFlagsEmpty(t *testing.T) {
	if got := formatCMakeFlags(nil); got != "" {
		t.Errorf("nil flags = %q, want empty", got)
	}
}
