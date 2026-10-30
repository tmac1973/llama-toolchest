package serverlog

import (
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

// What llama-server printed when a model loaded and then ran out of
// memory on its first request, taken from the machine this is from. The
// log held began in the middle of an earlier warm-up attempt, so the
// router's "lost the connection" line from that attempt comes first.
var crashLog = []string{
	"0.41.120.004 E srv    operator(): http client error: Failed to read connection",
	"0.41.120.010 I srv    operator(): instance name=big exited with status 1",
	"0.44.300.000 I srv          load: spawning server instance with name=big on port 50915",
	"[50915] 0.09.100.000 I slot launch_slot_: id  0 | task 0 | processing task",
	"[50915] /data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
	"[50915] 0.09.211.300 E CUDA error: out of memory",
	"[50915] 0.09.211.301 E   current device: 2, in function ggml_cuda_graph_evaluate_and_capture at /data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4209",
	"[50915] 0.09.211.302 E   cudaGraphLaunch(graph->instance, cuda_ctx->stream())",
	"[50915] /data/builds/b10448-cuda-optimized/libggml-base.so.0(ggml_abort+0x15b)[0x7fd7ee536c6b]",
	"[50915] /data/builds/b10448-cuda-optimized/libggml-cuda.so.0(_Z15ggml_cuda_errorPKcS0_S0_iS0_+0xb7)[0x7fd7e9a89927]",
	"[50915] /lib/x86_64-linux-gnu/libc.so.6(+0x29d90)[0x7fd7ed229d90]",
	"[50915] /data/builds/b10448-cuda-optimized/llama-server[0x55d0c1a2b3e5]",
}

func TestErrorLinesKeepsTheCauseFirst(t *testing.T) {
	got := ErrorLines(oomLog)
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
func TestErrorLinesDropsRepeatsFromRetries(t *testing.T) {
	twice := append(append([]string{}, oomLog...), oomLog...)
	if got, once := ErrorLines(twice), ErrorLines(oomLog); !reflect.DeepEqual(got, once) {
		t.Errorf("two attempts gave %d lines, one attempt gave %d", len(got), len(once))
	}
}

// A process that aborts prints straight to its error stream, with no
// timestamp and no level. Those lines are the only record of a crash.
func TestErrorLinesKeepsACrashThatBypassesTheLogger(t *testing.T) {
	got := ErrorLines([]string{
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

func TestErrorLinesIsBounded(t *testing.T) {
	var log []string
	for i := 0; i < 3*MaxErrorLines; i++ {
		log = append(log, "0.00.000.001 E srv  problem number "+strings.Repeat("x", i+1))
	}
	got := ErrorLines(log)
	if len(got) != MaxErrorLines {
		t.Fatalf("kept %d lines, want %d", len(got), MaxErrorLines)
	}
	if got[0] != "srv  problem number x" {
		t.Errorf("first kept line = %q, want the first one printed", got[0])
	}
}

// Backtrace frames are most of what an aborting process prints. They
// name no cause, and they would push the lines that do out of the dozen
// that are kept.
func TestErrorLinesLeavesOutBacktraceFrames(t *testing.T) {
	lines := ErrorLines(crashLog)
	for _, l := range lines {
		if strings.Contains(l, "[0x") {
			t.Errorf("kept a backtrace frame: %q", l)
		}
	}
	want := []string{
		"srv    operator(): http client error: Failed to read connection",
		"/data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:106: CUDA error",
		"CUDA error: out of memory",
		"current device: 2, in function ggml_cuda_graph_evaluate_and_capture at /data/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4209",
		"cudaGraphLaunch(graph->instance, cuda_ctx->stream())",
	}
	if !reflect.DeepEqual(lines, want) {
		t.Errorf("error lines =\n%s\nwant\n%s", strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

func TestParseSpawn(t *testing.T) {
	cases := []struct {
		line, name, port string
		ok               bool
	}{
		{"0.02.475.230 I srv          load: spawning server instance with name=unsloth--Qwen3.6-35B on port 45221", "unsloth--Qwen3.6-35B", "45221", true},
		// A model id may contain the words the line uses to delimit it.
		{"0.00.100.000 I srv          load: spawning server instance with name=odd on port name on port 46231", "odd on port name", "46231", true},
		{"0.02.475.263 I srv          load: spawning server instance with args:", "", "", false},
		{"[45221] 0.00.473.556 I cmn  common_param:   - CPU", "", "", false},
	}
	for _, c := range cases {
		name, port, ok := ParseSpawn(c.line)
		if name != c.name || port != c.port || ok != c.ok {
			t.Errorf("ParseSpawn(%q) = %q, %q, %v; want %q, %q, %v", c.line, name, port, ok, c.name, c.port, c.ok)
		}
	}
}

func TestReadyPort(t *testing.T) {
	cases := []struct {
		line, port string
		ok         bool
	}{
		{`[46231] cmd_child_to_router:state:{"state":"ready","payload":{}}`, "46231", true},
		// The router logs it at debug level under its own prefix.
		{`0.14.275.900 D [46231] cmd_child_to_router:state:{"state":"ready","payload":{"n_ctx":8192}}`, "46231", true},
		// The router pads a port below 10000.
		{`0.14.275.900 D [ 8081] cmd_child_to_router:state:{"state":"ready"}`, "8081", true},
		{`[46231] cmd_child_to_router:state:{"state":"loading"}`, "", false},
		{"0.00.100.000 I srv  router listening", "", false},
	}
	for _, c := range cases {
		port, ok := ReadyPort(c.line)
		if port != c.port || ok != c.ok {
			t.Errorf("ReadyPort(%q) = %q, %v; want %q, %v", c.line, port, ok, c.port, c.ok)
		}
	}
}

func TestInstancePort(t *testing.T) {
	cases := []struct {
		line, port string
		ok         bool
	}{
		{"[50915] 0.09.211.300 E CUDA error: out of memory", "50915", true},
		{"[ 8081] CUDA error: out of memory", "8081", true},
		{"0.41.120.010 I srv    operator(): instance name=big exited with status 1", "", false},
		{"==> Router stopped", "", false},
	}
	for _, c := range cases {
		port, ok := InstancePort(c.line)
		if port != c.port || ok != c.ok {
			t.Errorf("InstancePort(%q) = %q, %v; want %q, %v", c.line, port, ok, c.port, c.ok)
		}
	}
}
