package api

import (
	"encoding/csv"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
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
	big := make([]string, 30)
	for i := range big {
		big[i] = strings.Repeat("9", 1+i%3)
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
