package api

import (
	"bytes"
	"strings"
	"testing"
)

// A failed run's detail view shows what llama-server printed about the
// failure, with a tooltip that says how to read it.
func TestRunDetailShowsWhatTheServerReported(t *testing.T) {
	run := perfRun("r1")
	run.Status = "failed"
	run.Error = "warmup failed after retries: HTTP 500: model failed to load. llama-server reported: cudaMalloc failed: out of memory"
	run.FailureLog = []string{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 2: cudaMalloc failed: out of memory",
		"llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache",
	}
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_detail", &run); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()

	for _, want := range []string{
		"What llama-server reported (2 lines)",
		"allocating 5120.00 MiB on device 2",
		"failed to allocate buffer for kv cache",
		"The first line is usually the cause",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("detail view missing %q\n%s", want, out)
		}
	}
}

// A run that failed with nothing in the server log has no empty section.
func TestRunDetailOmitsTheServerReportWhenThereIsNone(t *testing.T) {
	run := perfRun("r1")
	run.Status = "failed"
	run.Error = "cancelled"
	var buf bytes.Buffer
	if err := benchTemplates(t).ExecuteTemplate(&buf, "benchmark_detail", &run); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if strings.Contains(buf.String(), "What llama-server reported") {
		t.Errorf("an empty server report is being shown\n%s", buf.String())
	}
}
