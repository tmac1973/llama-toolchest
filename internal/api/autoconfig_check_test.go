package api

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/autoconfig"
	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

func checkJob(status, errText, runID string) *benchmark.BenchmarkJob {
	return &benchmark.BenchmarkJob{Cells: []benchmark.JobCell{{Status: status, Error: errText, BenchmarkRunID: runID}}}
}

// A completed test load is a pass when there is no record of the cards
// running short.
func TestCheckFromJobPasses(t *testing.T) {
	chk := checkFromJob(checkJob(benchmark.CellStatusCompleted, "", "r1"), nil)
	if !chk.OK {
		t.Errorf("check = %+v, want OK", chk)
	}
}

// A test load that answered but left a card nearly full is treated as
// running out of memory on that card, so Autoconfigure adjusts it.
func TestCheckFromJobFlagsACardLeftNearlyFull(t *testing.T) {
	run := &benchmark.BenchmarkRun{Cards: []benchmark.CardMemory{
		{Index: 0, UsedMiB: 15879, TotalMiB: 16376},
		{Index: 1, UsedMiB: 14649, TotalMiB: 16376},
	}}
	get := func(string) (*benchmark.BenchmarkRun, error) { return run, nil }
	chk := checkFromJob(checkJob(benchmark.CellStatusCompleted, "", "r1"), get)
	if chk.OK || !chk.OutOfMemory || !chk.LowMemory || chk.Device != 0 {
		t.Errorf("check = %+v, want low memory on GPU 0", chk)
	}
	if !strings.Contains(chk.Reason, "497 MiB free on GPU 0") {
		t.Errorf("reason = %q, want it to say how much was left on which GPU", chk.Reason)
	}

	run.Cards[0].UsedMiB = 14000
	if chk := checkFromJob(checkJob(benchmark.CellStatusCompleted, "", "r1"), get); !chk.OK {
		t.Errorf("check = %+v, want OK with room on every card", chk)
	}
}

// The failure this was written for, as the run recorded it: the model
// loaded and ran out of memory on its first request. The cell's error
// quotes the out-of-memory line; the GPU is named two lines later, in
// the run's own log.
func TestCheckFromJobFindsTheGPUThatRanOut(t *testing.T) {
	run := &benchmark.BenchmarkRun{FailureLog: []string{
		"srv    operator(): http client error: Failed to read connection",
		"/data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
		"CUDA error: out of memory",
		"current device: 2, in function ggml_cuda_graph_evaluate_and_capture at /data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4209",
	}}
	get := func(string) (*benchmark.BenchmarkRun, error) { return run, nil }
	errText := "warmup failed after retries: HTTP 500: proxy error: Failed to read connection. llama-server reported: CUDA error: out of memory"

	chk := checkFromJob(checkJob(benchmark.CellStatusFailed, errText, "r1"), get)
	if chk.OK || !chk.OutOfMemory || chk.Device != 2 {
		t.Errorf("check = %+v, want out of memory on GPU 2", chk)
	}
	if chk.Reason != errText {
		t.Errorf("reason = %q, want the cell's error", chk.Reason)
	}
}

// A buffer that could not be allocated while loading names its GPU in
// the same line.
func TestCheckFromJobReadsTheGPUFromAnAllocationFailure(t *testing.T) {
	run := &benchmark.BenchmarkRun{FailureLog: []string{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 1: cudaMalloc failed: out of memory",
	}}
	get := func(string) (*benchmark.BenchmarkRun, error) { return run, nil }
	chk := checkFromJob(checkJob(benchmark.CellStatusFailed, "warmup failed after retries: HTTP 500", "r1"), get)
	if !chk.OutOfMemory || chk.Device != 1 {
		t.Errorf("check = %+v, want out of memory on GPU 1", chk)
	}
}

// A failure with nothing about memory in it is not one to step down for,
// and names no GPU.
func TestCheckFromJobLeavesOtherFailuresAlone(t *testing.T) {
	get := func(string) (*benchmark.BenchmarkRun, error) { return nil, errors.New("no such run") }
	chk := checkFromJob(checkJob(benchmark.CellStatusFailed, "build b1 no longer exists", ""), get)
	if chk.OK || chk.OutOfMemory || chk.Device != -1 {
		t.Errorf("check = %+v, want a plain failure with no GPU", chk)
	}
	if chk := checkFromJob(&benchmark.BenchmarkJob{}, get); chk.OK || chk.Reason == "" {
		t.Errorf("a job with no cells gave %+v, want a failure with a reason", chk)
	}
}

func TestCheckSummary(t *testing.T) {
	tests := []struct {
		name   string
		check  autoconfig.Verification
		ok     bool
		wants  []string
		absent []string
	}{
		{"no check", autoconfig.Verification{}, false, nil, nil},
		{"passed first time", autoconfig.Verification{Status: autoconfig.VerifyPassed, Attempts: 1}, true,
			[]string{"Checked", "answered a test request"}, []string{"adjusted"}},
		{"passed after one adjustment", autoconfig.Verification{Status: autoconfig.VerifyPassed, Attempts: 2, Adjusted: 1}, true,
			[]string{"ran out of GPU memory", "1 setting was adjusted"}, nil},
		{"passed after two", autoconfig.Verification{Status: autoconfig.VerifyPassed, Attempts: 3, Adjusted: 2}, true,
			[]string{"2 settings were adjusted"}, nil},
		{"failed", autoconfig.Verification{Status: autoconfig.VerifyFailed, Attempts: 5, Reason: "CUDA error: out of memory"}, false,
			[]string{"did not run", "5 test loads", "CUDA error: out of memory", "have not been made to work"}, nil},
		{"skipped", autoconfig.Verification{Status: autoconfig.VerifySkipped, Reason: "a benchmark started and is using the GPU"}, false,
			[]string{"Not checked", "a benchmark started", "have not been loaded"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, ok := checkSummary(tt.check)
			if ok != tt.ok {
				t.Errorf("ok = %v, want %v", ok, tt.ok)
			}
			if len(tt.wants) == 0 && text != "" {
				t.Errorf("text = %q, want none", text)
			}
			for _, w := range tt.wants {
				if !strings.Contains(text, w) {
					t.Errorf("text %q does not contain %q", text, w)
				}
			}
			for _, a := range tt.absent {
				if strings.Contains(text, a) {
					t.Errorf("text %q contains %q", text, a)
				}
			}
		})
	}
}

// The review shows the outcome of the test load with a tooltip that says
// how to read it, and shows nothing when no check ran.
func TestReviewShowsTheTestLoad(t *testing.T) {
	d := autoconfigReviewData{ModelID: "m", ModelName: "M", Fits: true, EstimateGiB: 40, BudgetGiB: 45,
		CheckHelp: autoconfigCheckHelp,
		Changed: []reviewRow{{Label: "GPU assignment", Current: "all GPUs", Proposed: "a custom split (20,20,17)",
			Why: []string{"A test load ran out of memory on GPU 2."}, Source: "test load"}}}
	d.Check, d.CheckOK = checkSummary(autoconfig.Verification{Status: autoconfig.VerifyPassed, Attempts: 2, Adjusted: 1})

	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "autoconfig_review", d); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	for _, want := range []string{"1 setting was adjusted", "sends the model one request",
		"a custom split (20,20,17)", "a test load (what ran when the settings were loaded)"} {
		if !strings.Contains(out, want) && !strings.Contains(out, strings.ReplaceAll(want, `"`, "&#34;")) {
			t.Errorf("review missing %q", want)
		}
	}

	d.Check, d.CheckOK = "", false
	buf.Reset()
	if err := benchTemplates(t).ExecuteTemplate(&buf, "autoconfig_review", d); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "Checked:") || strings.Contains(buf.String(), "Not checked") {
		t.Error("the review claims a check when none ran")
	}
}
