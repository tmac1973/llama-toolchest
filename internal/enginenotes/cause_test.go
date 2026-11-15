package enginenotes

import (
	"testing"
	"time"
)

// The lines are from real failures on the machines this was written for.
func TestCauseOf(t *testing.T) {
	cases := []struct {
		name string
		e    Event
		want Cause
	}{
		{"GPU fault after load", Event{Kind: KindCrash, ErrorLines: []string{
			"ggml_cuda_compute_forward: CONCAT failed",
			"CUDA error: an illegal memory access was encountered"}}, CauseGPUFault},
		{"misaligned address", Event{Kind: KindCrash, ErrorLines: []string{
			"CUDA error: misaligned address"}}, CauseGPUFault},
		{"KV cache at load", Event{Kind: KindLoadFailed, ErrorLines: []string{
			"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 2: cudaMalloc failed: out of memory",
			"llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache"}}, CauseOutOfVRAM},
		{"CUDA graph after load", Event{Kind: KindCrash, ErrorLines: []string{
			"/data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:113: CUDA error",
			"CUDA error: out of memory",
			"cudaGraphInstantiate(&graph->instances[idx].handle, cuda_graph, nullptr, nullptr, 0)"}}, CauseOutOfVRAM},
		{"failed assertion", Event{Kind: KindLoadFailed, ErrorLines: []string{
			"/data/llama.cpp/ggml/src/ggml-backend-meta.cpp:573: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed"}}, CauseUnsupported},
		{"system memory", Event{Kind: KindLoadFailed, ErrorLines: []string{
			"terminate called after throwing an instance of 'std::bad_alloc'"}}, CauseOutOfRAM},
		{"killed", Event{Kind: KindCrash, ExitStatus: 137}, CauseOutOfRAM},
		{"nothing to go on", Event{Kind: KindLoadFailed, ErrorLines: []string{
			"srv  load_model: failed to load model"}}, CauseOther},
		// "not supported" in an ordinary line of the tail is not the cause.
		{"word in the tail only", Event{Kind: KindCrash, ExitStatus: 1, Tail: []string{
			"I llama_context: flash_attn is not supported by the CPU backend"}}, CauseOther},
		{"not a failure", Event{Kind: KindUnload, ExitStatus: 137}, ""},

		// sd-server (stable-diffusion.cpp) on the RTX A4000s, 2026-10.
		{"sd-server CONCAT fault", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 134, ErrorLines: []string{
			"[E] ggml: ggml_cuda_compute_forward: CONCAT failed",
			"[E] ggml: CUDA error: an illegal memory access was encountered"}}, CauseGPUFault},
		// The progress bar can push the CUDA error line out; the failed
		// operation alone, with no memory error, is still a fault.
		{"sd-server failed operation only", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 134, Tail: []string{
			"  |======>   |  7/15 - 7.58it/s[E] ggml: ggml_cuda_compute_forward: CONCAT failed"}}, CauseGPUFault},
		// A real OOM: the decode did not fit, sd.cpp retried smaller, ran
		// out again and aborted.
		{"sd-server abort after a retry", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 134, Tail: []string{
			"[W] model manager cannot make enough memory available on CUDA0: need 25114.96 MB device / 24602.96 MB budget --- model_manager.cpp:2013",
			"[W] VAE decode ran out of memory; retrying with temporal tiling --- backend_fit.cpp:518",
			"[E] wan_vae segment 1/1 (graph) failed during weight preparation --- ggml_runner.cpp:898",
			"/data/stable-diffusion.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:111: CUDA error",
			"==> sd-server x exited with error: signal: aborted (core dumped)"}}, CauseOutOfVRAM},
		{"sd-server failed operation after a retry", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 134, Tail: []string{
			"[W] VAE decode ran out of memory; retrying with spatial+temporal tiling --- backend_fit.cpp:518",
			"[E] ggml: ggml_cuda_compute_forward: CONCAT failed",
			"[E] ggml: CUDA error: out of memory"}}, CauseOutOfVRAM},
		// A fault after a retry is still a fault: the retry got past the
		// memory problem.
		{"sd-server fault after a retry", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 134, Tail: []string{
			"[W] VAE decode ran out of memory; retrying with temporal tiling --- backend_fit.cpp:518",
			"[E] ggml: ggml_cuda_compute_forward: CONCAT failed",
			"[E] ggml: CUDA error: unspecified launch failure"}}, CauseGPUFault},
		// After a retry, a failure the later lines do not explain is the
		// memory problem the retry did not get past.
		{"sd-server nothing after a retry", Event{Source: SourceSDServer, Kind: KindCrash, ExitStatus: 1, ErrorLines: []string{
			"[W] VAE decode ran out of memory; retrying with temporal tiling",
			"[E] server: failed to write the response"}}, CauseOutOfVRAM},
		{"sd-server model manager", Event{Source: SourceSDServer, Kind: KindLoadFailed, ExitStatus: 134, ErrorLines: []string{
			"[W] model manager cannot make enough memory available on ROCm0: need 8092.58 MB device / 7580.58 MB budget"}}, CauseOutOfVRAM},
		{"ace-server killed", Event{Source: SourceAceServer, Kind: KindCrash, ExitStatus: 137}, CauseOutOfRAM},
	}
	for _, c := range cases {
		if got := CauseOf(c.e); got != c.want {
			t.Errorf("%s: cause = %q, want %q", c.name, got, c.want)
		}
	}
}

// Only what a job is there to find is expected: a setting that did not
// fit or was refused, during a job. A GPU fault never is, and nothing
// outside a job is.
func TestExpected(t *testing.T) {
	job := Event{Kind: KindCrash, JobID: "j"}
	user := Event{Kind: KindCrash}
	for _, c := range []struct {
		e     Event
		cause Cause
		want  bool
	}{
		{job, CauseOutOfVRAM, true},
		{job, CauseOutOfRAM, true},
		{job, CauseUnsupported, true},
		{job, CauseGPUFault, false},
		{job, CauseOther, false},
		{user, CauseOutOfVRAM, false},
	} {
		if got := Expected(c.e, c.cause); got != c.want {
			t.Errorf("Expected(job=%v, %s) = %v, want %v", c.e.JobID != "", c.cause, got, c.want)
		}
	}
}

// The summary counts each failure by cause and marks it, and never saves
// the cause with the event.
func TestSummaryCountsCauses(t *testing.T) {
	s, _, _, _ := testStore(t)
	s.Record(Event{Kind: KindCrash, ModelID: "a", JobID: "j", ErrorLines: []string{"CUDA error: out of memory"}})
	s.Record(Event{Kind: KindLoadFailed, ModelID: "a", JobID: "j", ErrorLines: []string{"GGML_ASSERT(x) failed"}})
	s.Record(Event{Kind: KindCrash, ModelID: "a", ErrorLines: []string{"CUDA error: an illegal memory access was encountered"}})

	sum := s.Summary(24*time.Hour, nil, nil)
	tot := sum.Totals
	if tot.OutOfVRAM != 1 || tot.Unsupported != 1 || tot.GPUFaults != 1 || tot.OtherFailures != 0 {
		t.Errorf("by cause = %+v", tot)
	}
	if tot.FailuresInJobs != 2 || tot.ExpectedFailures != 2 {
		t.Errorf("in jobs = %d, expected = %d, want 2 and 2", tot.FailuresInJobs, tot.ExpectedFailures)
	}
	for _, f := range sum.Failures {
		if f.Cause == "" {
			t.Errorf("failure without a cause: %+v", f)
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, e := range s.data.Events {
		if e.Cause != "" || e.Expected {
			t.Errorf("cause saved with the event: %+v", e)
		}
	}
}
