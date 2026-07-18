package api

import (
	"encoding/csv"
	"strconv"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func baseReq() jobCreateRequest {
	return jobCreateRequest{
		Name:     "job",
		ModelIDs: []string{"m"},
		BuildIDs: []string{"b"},
		Presets:  []string{"internal-quick"},
	}
}

// The form posts raw strings; the server owns parsing so the browser
// never reimplements value syntax.
func TestResolveSweepsParsesRawFormInput(t *testing.T) {
	req := baseReq()
	req.SweepsRaw = map[string]string{"gpu_layers": "20, 40,99"}

	if err := resolveSweeps(&req); err != nil {
		t.Fatalf("resolveSweeps: %v", err)
	}
	if len(req.Sweeps) != 1 {
		t.Fatalf("got %d axes, want 1", len(req.Sweeps))
	}
	if got := req.Sweeps[0].Values; len(got) != 3 || got[0] != "20" || got[2] != "99" {
		t.Errorf("values = %v, want [20 40 99]", got)
	}
}

// A blank field is an untouched form input, not an empty axis.
func TestResolveSweepsSkipsBlankFields(t *testing.T) {
	req := baseReq()
	req.SweepsRaw = map[string]string{"gpu_layers": "  ", "threads": "4,8"}

	if err := resolveSweeps(&req); err != nil {
		t.Fatalf("resolveSweeps: %v", err)
	}
	if len(req.Sweeps) != 1 || req.Sweeps[0].Field != "threads" {
		t.Errorf("got %+v, want only a threads axis", req.Sweeps)
	}
}

// JSON API callers can send structured axes directly; raw input must not
// clobber them.
func TestResolveSweepsPrefersStructuredAxes(t *testing.T) {
	req := baseReq()
	req.Sweeps = []benchmark.SweepAxis{{Field: "threads", Values: []string{"8"}}}
	req.SweepsRaw = map[string]string{"gpu_layers": "20,40"}

	if err := resolveSweeps(&req); err != nil {
		t.Fatalf("resolveSweeps: %v", err)
	}
	if len(req.Sweeps) != 1 || req.Sweeps[0].Field != "threads" {
		t.Errorf("structured axes were overwritten: %+v", req.Sweeps)
	}
}

func TestResolveSweepsRejectsBadValues(t *testing.T) {
	req := baseReq()
	req.SweepsRaw = map[string]string{"gpu_layers": "20,not-a-number"}

	if err := resolveSweeps(&req); err == nil {
		t.Error("expected an error for an unparseable value")
	}
}

// A bad sweep must fail when the job is defined, not hours into a run.
func TestValidateJobRequestRejectsInvalidSweep(t *testing.T) {
	req := baseReq()
	req.Sweeps = []benchmark.SweepAxis{{Field: "gpu_layers", Values: []string{"abc"}}}

	if err := validateJobRequest(req); err == nil {
		t.Error("expected validation to reject an unparseable sweep value")
	}
}

// Every axis multiplies, so a few long lists can silently queue days of
// work. The cap is checked against the expanded size, not the input size.
func TestValidateJobRequestCapsMatrixSize(t *testing.T) {
	req := baseReq()
	// Distinct values: duplicates are rejected earlier now, which would
	// mask the cap this test is about.
	big := make([]string, 30)
	for i := range big {
		big[i] = strconv.Itoa(i + 1)
	}
	req.Sweeps = []benchmark.SweepAxis{
		{Field: "gpu_layers", Values: big},
		{Field: "threads", Values: big},
	}

	err := validateJobRequest(req)
	if err == nil {
		t.Fatal("expected a runaway matrix to be rejected")
	}
	if !strings.Contains(err.Error(), "cells") {
		t.Errorf("error should explain the cell count, got: %v", err)
	}
}

func TestValidateJobRequestAcceptsReasonableSweep(t *testing.T) {
	req := baseReq()
	req.Sweeps = []benchmark.SweepAxis{
		{Field: "gpu_layers", Values: []string{"20", "40", "99"}},
	}
	if err := validateJobRequest(req); err != nil {
		t.Errorf("valid sweep rejected: %v", err)
	}
}

// Sweep points must be distinguishable in exports for the same reason
// cmake_flags had to be: otherwise the rows differ only by a number.
func TestCSVIncludesSweepColumn(t *testing.T) {
	runs, jobs := twoBuildRuns()
	runs[0].SweepValues = map[string]string{"gpu_layers": "40"}
	runs[1].SweepValues = map[string]string{"gpu_layers": "99"}

	rows := parseCSV(t, func(cw *csv.Writer) error {
		return writeCSVCells(cw, runs, jobs)
	})

	idx := columnIndex(t, rows[0], "sweep")
	if rows[1][idx] != "gpu_layers=40" {
		t.Errorf("row 1 sweep = %q, want gpu_layers=40", rows[1][idx])
	}
	if rows[2][idx] != "gpu_layers=99" {
		t.Errorf("row 2 sweep = %q, want gpu_layers=99", rows[2][idx])
	}
}

func TestFormatSweepValues(t *testing.T) {
	got := formatSweepValues(map[string]string{"threads": "8", "gpu_layers": "40"})
	if got != "gpu_layers=40 threads=8" {
		t.Errorf("got %q, want sorted \"gpu_layers=40 threads=8\"", got)
	}
	if formatSweepValues(nil) != "" {
		t.Error("nil should render empty")
	}
}

// batchMatrixServer builds a Server with a real registry holding one
// model, so the saved-config half of the check is exercised.
func batchMatrixServer(t *testing.T, saved models.ModelConfig) *Server {
	t.Helper()
	reg := models.NewRegistry(t.TempDir(), "/models")
	m := &models.Model{ID: "m", ModelID: "u/M", Quant: "Q8_0", FilePath: "/models/m.gguf"}
	if err := reg.Add(m); err != nil {
		t.Fatalf("add: %v", err)
	}
	saved.Enabled = true
	if err := reg.SetConfig("m", &saved); err != nil {
		t.Fatalf("set config: %v", err)
	}
	return &Server{registry: reg}
}

func intPtr(v int) *int { return &v }

// A micro-batch above the batch size must be refused when the job is
// defined. The apply-time check catches it too, but only after the sweep
// has been running — potentially for an hour.
func TestValidateBatchMatrixRejectsExplicitBadPair(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{})
	err := s.validateBatchMatrix([]string{"m"},
		&benchmark.ConfigOverrides{BatchSize: intPtr(512), UBatchSize: intPtr(4096)}, nil)
	if err == nil {
		t.Error("expected an explicitly bad batch pair to be rejected")
	}
}

// One bad point anywhere in the ladder fails the whole job up front,
// rather than that cell dying mid-run.
func TestValidateBatchMatrixRejectsBadSweepPoint(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{})
	err := s.validateBatchMatrix([]string{"m"},
		&benchmark.ConfigOverrides{BatchSize: intPtr(2048)},
		[]benchmark.SweepAxis{{Field: "ubatch_size", Values: []string{"512", "1024", "4096"}}})
	if err == nil {
		t.Error("expected the 4096 point to be rejected against batch 2048")
	}
}

// The saved-config half: sweeping only micro-batch has to be checked
// against whatever batch size the model already has.
func TestValidateBatchMatrixUsesSavedBatchSize(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{BatchSize: 1024})
	err := s.validateBatchMatrix([]string{"m"}, nil,
		[]benchmark.SweepAxis{{Field: "ubatch_size", Values: []string{"512", "2048"}}})
	if err == nil {
		t.Error("2048 exceeds the model's saved batch size of 1024 and should be rejected")
	}
}

// With no batch size saved, llama.cpp's 2048 default applies.
func TestValidateBatchMatrixUsesLlamaDefaultBatch(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{})
	if err := s.validateBatchMatrix([]string{"m"}, nil,
		[]benchmark.SweepAxis{{Field: "ubatch_size", Values: []string{"4096"}}}); err == nil {
		t.Error("4096 exceeds the default batch of 2048 and should be rejected")
	}
	if err := s.validateBatchMatrix([]string{"m"}, nil,
		[]benchmark.SweepAxis{{Field: "ubatch_size", Values: []string{"64", "512", "2048"}}}); err != nil {
		t.Errorf("a valid ladder was rejected: %v", err)
	}
}

// A job that doesn't touch batch sizes must not be validated against
// them at all.
func TestValidateBatchMatrixIgnoresUnrelatedJobs(t *testing.T) {
	s := batchMatrixServer(t, models.ModelConfig{BatchSize: 512})
	if err := s.validateBatchMatrix([]string{"m"}, nil,
		[]benchmark.SweepAxis{{Field: "threads", Values: []string{"4", "8"}}}); err != nil {
		t.Errorf("unrelated sweep rejected: %v", err)
	}
}
