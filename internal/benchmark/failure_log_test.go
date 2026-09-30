package benchmark

import (
	"context"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// A load that runs out of memory, as llama-server prints it: the
// allocation that did not fit, then each layer above reporting that it
// could not carry on. Info lines are mixed in, and the instance prefix is
// padded the way the router pads a port below 10000.
var oomLog = []string{
	"0.00.412.118 I srv  operator(): spawning server instance with name=big on port 40947",
	"[40947] 0.01.004.220 I load_tensors: offloaded 66/66 layers to GPU",
	"[40947] 0.03.551.902 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 2: cudaMalloc failed: out of memory",
	"[40947] 0.03.551.940 E alloc_tensor_range: failed to allocate CUDA2 buffer of size 5368709120",
	"[40947] 0.03.552.011 E llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache",
	"[40947] 0.03.552.300 W srv    load_model: retrying is not going to help",
	"[40947] 0.03.553.871 E main: exiting due to model loading error",
	"0.03.700.004 E srv  operator(): instance name=big exited with status 1",
}

func TestServerErrorLinesKeepsTheCauseFirst(t *testing.T) {
	got := serverErrorLines(oomLog)
	want := []string{
		"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5120.00 MiB on device 2: cudaMalloc failed: out of memory",
		"alloc_tensor_range: failed to allocate CUDA2 buffer of size 5368709120",
		"llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache",
		"main: exiting due to model loading error",
		"srv  operator(): instance name=big exited with status 1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("error lines =\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

// A warm-up is retried, and every retry loads the model and prints the
// same failure. The record should say it once.
func TestServerErrorLinesDropsRepeatsFromRetries(t *testing.T) {
	twice := append(append([]string{}, oomLog...), oomLog...)
	if got, once := serverErrorLines(twice), serverErrorLines(oomLog); !reflect.DeepEqual(got, once) {
		t.Errorf("two attempts gave %d lines, one attempt gave %d", len(got), len(once))
	}
}

// A process that aborts prints straight to its error stream, with no
// timestamp and no level. Those lines are the only record of a crash.
func TestServerErrorLinesKeepsACrashThatBypassesTheLogger(t *testing.T) {
	got := serverErrorLines([]string{
		"[ 8081] 0.09.100.000 I slot launch_slot_: id  0 | task 0 | processing task",
		"[ 8081] /src/ggml/src/ggml-cuda/ggml-cuda.cu:98: CUDA error",
		"[ 8081] CUDA error: out of memory",
		"[ 8081]   current device: 2, in function alloc at /src/ggml/src/ggml-cuda/ggml-cuda.cu:512",
		"==> Router stopped",
	})
	want := []string{
		"/src/ggml/src/ggml-cuda/ggml-cuda.cu:98: CUDA error",
		"CUDA error: out of memory",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("error lines = %q, want %q", got, want)
	}
}

func TestServerErrorLinesIsBounded(t *testing.T) {
	var log []string
	for i := 0; i < 3*maxFailureLogLines; i++ {
		log = append(log, "0.00.000.001 E srv  problem number "+strings.Repeat("x", i+1))
	}
	got := serverErrorLines(log)
	if len(got) != maxFailureLogLines {
		t.Fatalf("kept %d lines, want %d", len(got), maxFailureLogLines)
	}
	if got[0] != "srv  problem number x" {
		t.Errorf("first kept line = %q, want the first one printed", got[0])
	}
}

// Only what the server printed after the run started belongs to the run.
// An error left in the log by whatever ran before it must not be blamed
// on this one.
func TestLinesAfterTakesOnlyTheRunsOwnOutput(t *testing.T) {
	log := []string{"a", "b", "c", "d"}
	if got := linesAfter(log, "b"); !reflect.DeepEqual(got, []string{"c", "d"}) {
		t.Errorf("after b = %q", got)
	}
	if got := linesAfter(log, ""); len(got) != 4 {
		t.Errorf("an empty mark kept %d of 4 lines", len(got))
	}
	// The mark has been pushed out of the buffer: everything still held
	// was printed after it.
	if got := linesAfter(log, "gone"); len(got) != 4 {
		t.Errorf("a mark no longer held kept %d of 4 lines", len(got))
	}
}

func TestFailureHeadlineIgnoresTheQuotedLine(t *testing.T) {
	const head = "warmup failed after retries: HTTP 500: model failed to load"
	layer := withServerReason(head, []string{"allocating 5120.00 MiB on device 2: cudaMalloc failed: out of memory"})
	tensor := withServerReason(head, []string{"allocating 3072.00 MiB on device 0: cudaMalloc failed: out of memory"})

	if layer == tensor {
		t.Fatal("test is not comparing two different errors")
	}
	if failureHeadline(layer) != head || failureHeadline(tensor) != head {
		t.Errorf("headlines = %q and %q, want %q", failureHeadline(layer), failureHeadline(tensor), head)
	}
	if got := withServerReason(head, nil); got != head {
		t.Errorf("with nothing reported the error became %q", got)
	}
}

// End to end through Runner.Run: a load the router refuses must leave the
// run with what the server printed about it, and not with what was in the
// log before the run began.
func TestFailedRunRecordsWhatTheServerReported(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[]`))
	})
	mux.HandleFunc("/models/load", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":{"message":"model name=big failed to load"}}`, http.StatusInternalServerError)
	})
	router := httptest.NewServer(mux)
	defer router.Close()

	before := []string{
		"0.00.100.000 E srv  an error from an earlier run",
		"0.00.200.000 I srv  router listening",
	}
	calls := 0
	serverLog := func() []string {
		calls++
		if calls == 1 {
			return before
		}
		return append(append([]string{}, before...), oomLog...)
	}

	store := NewStore(t.TempDir(), nil)
	run := BenchmarkRun{ID: "run-1", ModelID: "big", Status: StatusRunning}
	store.Save(run)
	NewRunner(store).Run(context.Background(), RunConfig{
		Run:        run,
		Preset:     GetPreset("internal-quick"),
		RouterURL:  router.URL,
		RouterName: "big",
		ServerLog:  serverLog,
	}, nil)

	got, err := store.Get("run-1")
	if err != nil {
		t.Fatalf("read back run: %v", err)
	}
	if got.Status != StatusFailed {
		t.Fatalf("status = %s, want failed", got.Status)
	}
	if !strings.HasPrefix(got.Error, "failed to load model: HTTP 500") {
		t.Errorf("error = %q, want it to start with the run's own error", got.Error)
	}
	if !strings.Contains(got.Error, "cudaMalloc failed: out of memory") {
		t.Errorf("error = %q, want it to quote the cause from the server log", got.Error)
	}
	if len(got.FailureLog) == 0 || !strings.Contains(got.FailureLog[0], "out of memory") {
		t.Errorf("failure log = %q, want the out-of-memory line first", got.FailureLog)
	}
	for _, line := range got.FailureLog {
		if strings.Contains(line, "earlier run") {
			t.Errorf("failure log includes a line printed before the run started: %q", line)
		}
	}
}
