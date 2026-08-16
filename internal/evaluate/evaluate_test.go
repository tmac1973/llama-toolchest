package evaluate

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

// writeFakeBinary writes an executable shell script standing in for
// llama-perplexity: it logs its argv and LD_LIBRARY_PATH (when
// logPath != ""), runs body, and exits with exitStatus.
func writeFakeBinary(t *testing.T, dir, body, logPath string, exitStatus int) string {
	t.Helper()
	script := "#!/bin/sh\n"
	if logPath != "" {
		script += "echo \"$@\" > '" + logPath + "'\n"
		script += "echo \"$LD_LIBRARY_PATH\" >> '" + logPath + "'\n"
	}
	script += body + "exit " + strconv.Itoa(exitStatus) + "\n"
	bin := filepath.Join(dir, "llama-perplexity")
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

// ---------- flag mapping ----------

func TestMapConfigFlagsFull(t *testing.T) {
	snap := SnapshotSubset{
		GPULayers:      33,
		Threads:        8,
		BatchSize:      512,
		UBatchSize:     128,
		FlashAttention: true,
		KVCacheQuant:   "q8_0",
		DirectIO:       true,
		PlacementFlags: []string{"--device", "ROCm0,ROCm1", "--tensor-split", "1,1"},
	}
	got := MapConfigFlags(snap)
	want := []string{
		"--n-gpu-layers", "33",
		"--threads", "8",
		"--batch-size", "512",
		"--ubatch-size", "128",
		"--flash-attn", "on",
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"--direct-io",
		"--device", "ROCm0,ROCm1",
		"--tensor-split", "1,1",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MapConfigFlags = %v, want %v", got, want)
	}
}

// KV quant must reach BOTH cache-type flags, and a GPU subset must
// emit the device list the placement resolver produced.
func TestMapConfigFlagsKVQuantAndDeviceList(t *testing.T) {
	got := strings.Join(MapConfigFlags(SnapshotSubset{
		GPULayers:      999,
		Threads:        4,
		KVCacheQuant:   "q4_0",
		PlacementFlags: []string{"--device", "Vulkan0,Vulkan2"},
	}), " ")
	for _, want := range []string{"--cache-type-k q4_0", "--cache-type-v q4_0", "--device Vulkan0,Vulkan2"} {
		if !strings.Contains(got, want) {
			t.Errorf("flags missing %q: %s", want, got)
		}
	}
}

// Excluded parameters must provably be ABSENT: the allow-list is the
// complete statement, so every excluded knob is asserted by name.
func TestMapConfigFlagsExcludesAreAbsent(t *testing.T) {
	got := strings.Join(MapConfigFlags(SnapshotSubset{
		GPULayers:      33,
		Threads:        8,
		BatchSize:      2048,
		UBatchSize:     512,
		FlashAttention: true,
		KVCacheQuant:   "q8_0",
		DirectIO:       true,
	}), " ")
	for _, excluded := range []string{
		"--ctx-size", // model context replaced by the fixed EvalContextSize
		"--context-size",
		"--parallel",  // parallel slots
		"--spec-type", // speculative decoding
		"--model-draft",
		"--ngram",
		"--temperature", // sampling
		"--top-p",
		"--mmproj", // vision projector
		"--context-shift",
		"--split-mode", // placement without PlacementFlags
		"--main-gpu",
		"--tensor-split",
	} {
		if strings.Contains(got, excluded) {
			t.Errorf("excluded flag %q leaked into the mapping: %s", excluded, got)
		}
	}
}

// Zero batch/ubatch means "don't emit" (the tool keeps its defaults),
// and flash-attn is always pinned explicitly.
func TestMapConfigFlagsZeroConventions(t *testing.T) {
	got := strings.Join(MapConfigFlags(SnapshotSubset{GPULayers: 0, Threads: 0, FlashAttention: false}), " ")
	for _, want := range []string{"--n-gpu-layers 0", "--threads 0", "--flash-attn off"} {
		if !strings.Contains(got, want) {
			t.Errorf("flags missing %q: %s", want, got)
		}
	}
	for _, absent := range []string{"--batch-size", "--ubatch-size", "--cache-type-k", "--direct-io"} {
		if strings.Contains(got, absent) {
			t.Errorf("unset field %q should not be emitted: %s", absent, got)
		}
	}
}

// ---------- command assembly ----------

func TestAssembledArgsPerMode(t *testing.T) {
	flags := []string{"--n-gpu-layers", "33", "--threads", "8"}
	cases := []struct {
		spec Spec
		want []string
	}{
		{
			Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModePerplexity, DatasetPath: "wiki.txt", Chunks: 100, Flags: flags},
			[]string{"--model", "m.gguf", "--n-gpu-layers", "33", "--threads", "8",
				"--ctx-size", "512", "-f", "wiki.txt", "--chunks", "100"},
		},
		{
			Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModePerplexity, DatasetPath: "wiki.txt", Flags: flags},
			[]string{"--model", "m.gguf", "--n-gpu-layers", "33", "--threads", "8",
				"--ctx-size", "512", "-f", "wiki.txt"},
		},
		{
			Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeHellaSwag, DatasetPath: "hs.txt", Tasks: 400, Flags: flags},
			[]string{"--model", "m.gguf", "--n-gpu-layers", "33", "--threads", "8",
				"--ctx-size", "512", "--hellaswag", "-f", "hs.txt", "--hellaswag-tasks", "400"},
		},
		{
			Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeWinogrande, DatasetPath: "wg.csv", Flags: flags},
			[]string{"--model", "m.gguf", "--n-gpu-layers", "33", "--threads", "8",
				"--ctx-size", "512", "--winogrande", "-f", "wg.csv"},
		},
		{
			Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeKLDiv, DatasetPath: "wiki.txt", KLBasePath: "ref.kld", Flags: flags},
			[]string{"--model", "m.gguf", "--n-gpu-layers", "33", "--threads", "8",
				"--ctx-size", "512", "--kl-divergence", "--kl-divergence-base", "ref.kld", "-f", "wiki.txt"},
		},
	}
	for i, tc := range cases {
		got, err := tc.spec.args()
		if err != nil {
			t.Fatalf("case %d: args: %v", i, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("case %d:\n got  %v\n want %v", i, got, tc.want)
		}
	}
}

// The KL comparison must NOT carry --chunks: the chunk count comes
// solely from the base file, so the flag would be inert.
func TestKLComparisonOmitsChunks(t *testing.T) {
	spec := Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeKLDiv, DatasetPath: "wiki.txt",
		KLBasePath: "ref.kld", Chunks: 100, Flags: []string{"--threads", "8"}}
	got, err := spec.args()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.Join(got, " "), "--chunks") {
		t.Errorf("KL comparison must not pass --chunks (inert there): %v", got)
	}
}

func TestAssembledArgsValidation(t *testing.T) {
	cases := []struct {
		name string
		spec Spec
		want string
	}{
		{"no binary", Spec{ModelPath: "m.gguf", Mode: ModePerplexity, DatasetPath: "w.txt"}, "binary"},
		{"no model", Spec{Binary: "p", Mode: ModePerplexity, DatasetPath: "w.txt"}, "model"},
		{"no dataset", Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModePerplexity}, "dataset"},
		{"KL without base", Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeKLDiv, DatasetPath: "w.txt"}, "KLBasePath"},
		{"unknown mode", Spec{Binary: "p", ModelPath: "m.gguf", Mode: "nope", DatasetPath: "w.txt"}, "mode"},
	}
	for _, tc := range cases {
		if _, err := tc.spec.args(); err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%s: want error naming %q, got %v", tc.name, tc.want, err)
		}
	}
}

// KL base generation: same shape as the comparison minus
// --kl-divergence, plus an optional --chunks cap, and KLBasePath as the
// OUTPUT destination.
func TestKLBaseArgs(t *testing.T) {
	spec := Spec{Binary: "p", ModelPath: "m.gguf", Mode: ModeKLDiv, DatasetPath: "wiki.txt",
		KLBasePath: "out.kld", Chunks: 100, Flags: []string{"--threads", "8"}}
	got, err := spec.klBaseArgs()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--model", "m.gguf", "--threads", "8",
		"--ctx-size", "512", "--kl-divergence-base", "out.kld", "-f", "wiki.txt", "--chunks", "100"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("klBaseArgs = %v, want %v", got, want)
	}
	joined := strings.Join(got, " ")
	if strings.Contains(joined, "--kl-divergence ") || strings.Contains(joined, "--kl-divergence-base out.kld -f wiki.txt --kl-divergence") {
		t.Errorf("generation must not pass the --kl-divergence flag: %v", got)
	}

	if _, err := (Spec{Binary: "p", ModelPath: "m.gguf", DatasetPath: "wiki.txt"}).klBaseArgs(); err == nil {
		t.Error("generation without KLBasePath must fail")
	}
}

// ---------- execution ----------

func TestRunSuccessParsesCombinedOutput(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "argv.txt")
	// The score line goes to STDERR (LOG_INF), the rest to stdout — the
	// combined buffer must carry both.
	bin := writeFakeBinary(t, dir, "echo 'perplexity: calculating perplexity over 100 chunks, n_ctx=512' \\\n  1>&2\necho '[1]9.8765,'\necho 'Final estimate: PPL = 9.0123 +/- 0.00456' 1>&2\n", logPath, 0)

	res, err := Run(context.Background(), Spec{
		Binary: bin, ModelPath: "/models/m.gguf", Mode: ModePerplexity,
		DatasetPath: "/data/wiki.txt", Chunks: 100,
		Flags: []string{"--n-gpu-layers", "33", "--threads", "8"},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Perplexity != 9.0123 || res.PerplexityErr != 0.00456 {
		t.Errorf("scores = %v ± %v, want 9.0123 ± 0.00456", res.Perplexity, res.PerplexityErr)
	}
	if res.Mode != "perplexity" || res.Dataset != "wikitext-2" || res.ContextSize != 512 || res.Chunks != 100 {
		t.Errorf("result identity fields wrong: %+v", res)
	}

	// The recorded command line: model first, mapped flags, then the
	// fixed ctx and mode flags.
	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	wantArgs := "--model /models/m.gguf --n-gpu-layers 33 --threads 8 --ctx-size 512 -f /data/wiki.txt --chunks 100"
	if lines[0] != wantArgs {
		t.Errorf("argv = %q, want %q", lines[0], wantArgs)
	}
	// The binary's directory is prepended to LD_LIBRARY_PATH so the
	// co-located libllama.so / libggml*.so resolve.
	if !strings.Contains(lines[len(lines)-1], dir) {
		t.Errorf("LD_LIBRARY_PATH missing the binary dir %s: %q", dir, lines[len(lines)-1])
	}
	if idx := strings.Index(lines[len(lines)-1], dir); idx != 0 {
		t.Errorf("binary dir must be PREPENDED to LD_LIBRARY_PATH, got %q", lines[len(lines)-1])
	}
}

func TestRunNonZeroExitCarriesTail(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeBinary(t, dir, "echo 'loading model ..' 1>&2\necho 'error: failed to load /models/m.gguf' 1>&2\n", "", 1)

	_, err := Run(context.Background(), Spec{
		Binary: bin, ModelPath: "/models/m.gguf", Mode: ModePerplexity, DatasetPath: "/data/wiki.txt",
	})
	if err == nil {
		t.Fatal("want error on nonzero exit")
	}
	if !strings.Contains(err.Error(), "failed to load /models/m.gguf") {
		t.Errorf("error must carry the tail of the combined output, got: %v", err)
	}
	if !strings.Contains(err.Error(), "exit status 1") {
		t.Errorf("error should name the exit status, got: %v", err)
	}
}

func TestRunContextCancellationKillsProcess(t *testing.T) {
	dir := t.TempDir()
	// sleep's fds go to /dev/null so killing the script closes the
	// pipes immediately (an orphaned sleep holding them would stall
	// the I/O copies and fake a hang).
	bin := writeFakeBinary(t, dir, "sleep 30 >/dev/null 2>&1\n", "", 0)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := Run(ctx, Spec{
		Binary: bin, ModelPath: "/models/m.gguf", Mode: ModePerplexity, DatasetPath: "/data/wiki.txt",
	})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("want error on cancellation")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("error should wrap context.DeadlineExceeded for errors.Is, got: %v", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("cancellation took %v — the process was not killed", elapsed)
	}
}

func TestRunSpecValidationBeforeExec(t *testing.T) {
	_, err := Run(context.Background(), Spec{Binary: "/nonexistent/llama-perplexity", Mode: ModePerplexity})
	if err == nil || !strings.Contains(err.Error(), "model path") {
		t.Errorf("want spec validation error before any exec, got: %v", err)
	}
}

// ---------- GenerateKLBase ----------

func TestGenerateKLBaseRunsGenerationForm(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "argv.txt")
	bin := writeFakeBinary(t, dir, "echo 'saving all logits to out.kld' 1>&2\necho '[1]9.8765,' 1>&2\n", logPath, 0)

	err := GenerateKLBase(context.Background(), Spec{
		Binary: bin, ModelPath: "/models/m.gguf", Mode: ModeKLDiv,
		DatasetPath: "/data/wiki.txt", KLBasePath: filepath.Join(dir, "out.kld"), Chunks: 100,
	})
	if err != nil {
		t.Fatalf("GenerateKLBase: %v", err)
	}

	raw, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	want := "--model /models/m.gguf --ctx-size 512 --kl-divergence-base " + filepath.Join(dir, "out.kld") + " -f /data/wiki.txt --chunks 100"
	if lines[0] != want {
		t.Errorf("argv = %q, want %q", lines[0], want)
	}
	if strings.Contains(lines[0], "--kl-divergence ") || strings.HasSuffix(lines[0], "--kl-divergence") {
		t.Errorf("generation argv must not carry the --kl-divergence flag: %q", lines[0])
	}
}

func TestGenerateKLBaseFailureCarriesTail(t *testing.T) {
	dir := t.TempDir()
	bin := writeFakeBinary(t, dir, "echo 'error: could not write out.kld' 1>&2\n", "", 2)

	err := GenerateKLBase(context.Background(), Spec{
		Binary: bin, ModelPath: "/models/m.gguf", Mode: ModeKLDiv,
		DatasetPath: "/data/wiki.txt", KLBasePath: filepath.Join(dir, "out.kld"),
	})
	if err == nil || !strings.Contains(err.Error(), "could not write out.kld") {
		t.Errorf("want failure with output tail, got: %v", err)
	}
}

// ---------- tail buffer ----------

func TestTailBufferKeepsLastLimitBytes(t *testing.T) {
	tb := newTailBuffer(16)
	if _, err := tb.Write([]byte("0123456789")); err != nil {
		t.Fatal(err)
	}
	if _, err := tb.Write([]byte("abcdefghij")); err != nil {
		t.Fatal(err)
	}
	if got := tb.String(); got != "456789abcdefghij" {
		t.Errorf("tail = %q, want the last 16 bytes %q", got, "456789abcdefghij")
	}
	n, err := newTailBuffer(4).Write([]byte(""))
	if err != nil || n != 0 {
		t.Errorf("empty write should report (0, nil), got (%d, %v)", n, err)
	}
}
