// Package evaluate runs the llama-perplexity tool against a model and
// parses its output into typed capability scores: perplexity, KL
// divergence against reference logits, HellaSwag, and Winogrande.
//
// The engine is deliberately self-contained. It takes a Spec (binary,
// model path, mode, dataset path, task/chunk caps, pre-mapped flags)
// and returns a Result. It knows no registry, no store, and no API —
// the job runner resolves models and datasets, builds the Spec, and
// copies the Result onto the run's EvalScores. This package must not
// import internal/benchmark or internal/api: Result mirrors
// benchmark.EvalScores and MapConfigFlags takes a plain-field
// SnapshotSubset for exactly that reason.
package evaluate

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// Mode names the capability evaluation to run. The string values double
// as the value stored on the run's EvalScores.Mode.
type Mode string

const (
	ModePerplexity Mode = "perplexity"
	ModeKLDiv      Mode = "kl-divergence"
	ModeHellaSwag  Mode = "hellaswag"
	ModeWinogrande Mode = "winogrande"
)

// DatasetName returns the canonical name of the dataset the mode runs
// over. The mode fixes the dataset — perplexity and KL both score on
// the wikitext-2 test set, the accuracy modes each on their own — so
// the engine can name it without the runner. The pinned dataset table
// (phase 02) uses the same names.
func (m Mode) DatasetName() string {
	switch m {
	case ModePerplexity, ModeKLDiv:
		return "wikitext-2"
	case ModeHellaSwag:
		return "hellaswag"
	case ModeWinogrande:
		return "winogrande"
	}
	return ""
}

// EvalContextSize is the fixed context size every evaluation runs at
// (--ctx-size). The ctx value IS the perplexity chunk size
// (n_chunk_max = tokens/n_ctx, perplexity.cpp:493), so it determines
// the score; llama.cpp's own wikitext-2 perplexity figures are quoted
// at 512. The engine never relies on the tool's upstream default (which
// happens to be the same value, perplexity.cpp:2015, and a non-positive
// value is rejected outright, perplexity.cpp:2024-2030): it always
// passes the constant explicitly. It is also part of the KL cache key
// (phase 02).
const EvalContextSize = 512

// Spec describes one evaluation run. It is engine-shaped on purpose: no
// model ID, no snapshot — the api layer resolves those into concrete
// paths and flags (MapConfigFlags) before calling Run, so the recorded
// config and the flags that ran share one source.
type Spec struct {
	Binary      string // path to llama-perplexity
	ModelPath   string // path to the GGUF under evaluation
	Mode        Mode
	DatasetPath string // -f input (wikitext-2 test, hellaswag_val, winogrande csv)
	Tasks       int    // hellaswag/winogrande task cap; 0 = full
	Chunks      int    // perplexity/KL chunk cap; 0 = full
	// KLBasePath is the reference logits file: the input for ModeKLDiv
	// comparisons, and the output destination when GenerateKLBase writes
	// a fresh base.
	KLBasePath string
	// Flags are the pre-mapped model/config flags (MapConfigFlags).
	Flags []string
}

// Result is one evaluated outcome, and the ONE declaration of the
// capability-score schema: benchmark.EvalScores is an alias for it, so
// the run's stored JSON and the engine's output are the same type and
// cannot drift. It lives here rather than in internal/benchmark because
// this package must not import that one.
//
// The engine fills every field except Reference and ReferenceLabel:
// the runner records the KL reference's identity there.
type Result struct {
	Mode        string `json:"mode"`
	Dataset     string `json:"dataset,omitempty"`
	ContextSize int    `json:"context_size,omitempty"`
	// Chunks is the chunk count the tool announced it scored
	// (perplexity/KL), falling back to the requested cap when the
	// output carries no count. 0 means neither was known.
	Chunks int `json:"chunks,omitempty"`

	// Perplexity mode.
	Perplexity    float64 `json:"perplexity,omitempty"`
	PerplexityErr float64 `json:"perplexity_err,omitempty"`

	// HellaSwag / Winogrande modes. Accuracy is a percentage; the CI
	// bounds are asymmetric as printed by HellaSwag and
	// symmetric-derived (acc ± err) for Winogrande.
	Accuracy       float64 `json:"accuracy,omitempty"`
	AccuracyCILow  float64 `json:"accuracy_ci_low,omitempty"`
	AccuracyCIHigh float64 `json:"accuracy_ci_high,omitempty"`
	// Tasks is the number of tasks actually scored, as parsed from the
	// tool output.
	Tasks int `json:"tasks,omitempty"`

	// KL divergence mode (comparison against the reference logits).
	KLMean        float64 `json:"kl_mean,omitempty"`
	KLMeanErr     float64 `json:"kl_mean_err,omitempty"`
	KLMax         float64 `json:"kl_max,omitempty"`
	KLP999        float64 `json:"kl_p999,omitempty"`
	SameTopPct    float64 `json:"same_top_pct,omitempty"`
	SameTopPctErr float64 `json:"same_top_pct_err,omitempty"`

	// Reference is the registry ID of the model the KL divergence was
	// measured against — the stable identity, used for lookups.
	// ReferenceLabel is the same model written for a person to read
	// ("Qwen3.5-9B (IQ4_NL)"); displays prefer it and fall back to
	// Reference for runs recorded before it existed. Both are filled by
	// the runner and empty for other modes.
	Reference      string `json:"reference,omitempty"`
	ReferenceLabel string `json:"reference_label,omitempty"`
}

// SnapshotSubset is the plain-field view of benchmark.ConfigSnapshot
// that the flag mapping needs. MapConfigFlags takes this struct instead
// of the snapshot itself to avoid an import cycle; the api layer fills
// it from the snapshot plus the resolved GPU placement.
type SnapshotSubset struct {
	GPULayers      int
	Threads        int
	BatchSize      int
	UBatchSize     int
	FlashAttention bool
	KVCacheQuant   string
	DirectIO       bool
	// PlacementFlags are the GPU placement flags (--split-mode,
	// --main-gpu, --device, trimmed --tensor-split) formatted by the
	// caller via models.GPUPlacementFlags. Placement is NOT derivable
	// from the snapshot: it carries neither SplitMode nor MainGPU, and a
	// swept gpu_assign only becomes a tensor-split through the
	// merge-and-resolve the performance path already does.
	PlacementFlags []string
}

// MapConfigFlags renders the config fields that legitimately reach the
// evaluation command line. It is the SINGLE allow-list shared by
// execution (Spec.Flags) and the recorded snapshot, deciding what runs.
//
// Mapped from SnapshotSubset: --n-gpu-layers, --threads, --batch-size,
// --ubatch-size, --flash-attn on|off, --cache-type-k/-v (KV quant),
// --direct-io, then the caller's PlacementFlags. --batch-size /
// --ubatch-size follow the ModelConfig convention (zero = don't emit,
// the tool keeps its own defaults); --flash-attn is always emitted so
// the evaluation pins the state explicitly instead of inheriting the
// tool's "auto" default. --direct-io is accepted by the tool
// (common/arg.cpp:2656; marked DEPRECATED there — warning only, still
// valid); mapping it keeps the excluded list a complete statement.
//
// Excluded by listing — every ConfigSnapshot field is either mapped or
// named here:
//   - ContextSize: the model-config context size is replaced by the
//     fixed evaluation context, --ctx-size EvalContextSize (the command
//     assembly below), which is score-determining and comparable.
//   - GPUAssign / TensorSplit: not derivable from the snapshot (no
//     SplitMode / MainGPU); they reach the command line only as the
//     resolved PlacementFlags.
//   - SpecType, DraftModelPath, DraftMax, DraftMin, DraftPMin,
//     NgramSizeN, NgramSizeM: speculative decoding — scoring is a fixed
//     greedy next-token pass over a fixed corpus, there is nothing for
//     a drafter to accelerate.
//   - parallel slots, sampling, mmproj, context-shift (ModelConfig
//     fields the snapshot never carries): irrelevant to or incompatible
//     with single-stream deterministic scoring.
func MapConfigFlags(snap SnapshotSubset) []string {
	var flags []string
	flags = append(flags, "--n-gpu-layers", strconv.Itoa(snap.GPULayers))
	flags = append(flags, "--threads", strconv.Itoa(snap.Threads))
	if snap.BatchSize > 0 {
		flags = append(flags, "--batch-size", strconv.Itoa(snap.BatchSize))
	}
	if snap.UBatchSize > 0 {
		flags = append(flags, "--ubatch-size", strconv.Itoa(snap.UBatchSize))
	}
	if snap.FlashAttention {
		flags = append(flags, "--flash-attn", "on")
	} else {
		flags = append(flags, "--flash-attn", "off")
	}
	if snap.KVCacheQuant != "" {
		flags = append(flags, "--cache-type-k", snap.KVCacheQuant, "--cache-type-v", snap.KVCacheQuant)
	}
	if snap.DirectIO {
		flags = append(flags, "--direct-io")
	}
	flags = append(flags, snap.PlacementFlags...)
	return flags
}

// args builds the llama-perplexity argument list for a Run, documented
// against tools/perplexity/perplexity.cpp:
//
//	perplexity:     --model <m> <Flags> --ctx-size 512 -f <wikitext> [--chunks N]
//	hellaswag:      --model <m> <Flags> --ctx-size 512 --hellaswag -f <hellaswag_val> [--hellaswag-tasks N]
//	winogrande:     --model <m> <Flags> --ctx-size 512 --winogrande -f <winogrande.csv> [--winogrande-tasks N]
//	kl-divergence:  --model <m> <Flags> --ctx-size 512 --kl-divergence --kl-divergence-base <base.kld> -f <wikitext>
//
// Every invocation carries --ctx-size EvalContextSize — required by the
// tool (a non-positive value is rejected, perplexity.cpp:2024-2030) and
// score-determining (the ctx is the chunk size, perplexity.cpp:493).
//
// The KL comparison deliberately omits --chunks: the comparison's chunk
// count comes solely from the base file (params.n_chunks is read only
// by the perplexity functions, perplexity.cpp:344,495), so the flag
// would be inert. The base file embeds its own n_ctx, and the tool does
// NOT reliably guard a mismatch (perplexity.cpp:1717-1722 logs without
// returning; a smaller embedded n_ctx passes silently and wins) — the
// phase 02 cache key carrying ctx and chunks is the ONLY thing keeping
// base and comparison consistent.
func (s Spec) args() ([]string, error) {
	args, err := s.baseArgs()
	if err != nil {
		return nil, err
	}

	switch s.Mode {
	case ModePerplexity:
		args = append(args, "-f", s.DatasetPath)
		if s.Chunks > 0 {
			args = append(args, "--chunks", strconv.Itoa(s.Chunks))
		}
	case ModeHellaSwag:
		args = append(args, "--hellaswag", "-f", s.DatasetPath)
		if s.Tasks > 0 {
			args = append(args, "--hellaswag-tasks", strconv.Itoa(s.Tasks))
		}
	case ModeWinogrande:
		args = append(args, "--winogrande", "-f", s.DatasetPath)
		if s.Tasks > 0 {
			args = append(args, "--winogrande-tasks", strconv.Itoa(s.Tasks))
		}
	case ModeKLDiv:
		if s.KLBasePath == "" {
			return nil, fmt.Errorf("kl-divergence evaluation needs KLBasePath (the reference logits file)")
		}
		args = append(args, "--kl-divergence", "--kl-divergence-base", s.KLBasePath, "-f", s.DatasetPath)
	default:
		return nil, fmt.Errorf("unknown evaluation mode %q", s.Mode)
	}
	return args, nil
}

// klBaseArgs builds the argument list that GENERATES a KL base logits
// file: the comparison's flags minus --kl-divergence, plus an optional
// --chunks cap. Without the --kl-divergence flag the tool writes the
// base file to KLBasePath (perplexity.cpp:461-470).
//
// The execution model is the same as Run's (combined buffer, parsed
// after exit) but there is NO live output parsing: the per-chunk
// generation output is newline-less comma fragments
// (perplexity.cpp:633), so live progress cannot come from the stream.
// Progress during generation is the CALLER's job, by polling the
// growing output file's size against the phase 02 estimate (phase 03);
// the engine only runs the process.
func (s Spec) klBaseArgs() ([]string, error) {
	args, err := s.baseArgs()
	if err != nil {
		return nil, err
	}
	if s.KLBasePath == "" {
		return nil, fmt.Errorf("KL base generation needs KLBasePath (the output logits file)")
	}
	args = append(args, "--kl-divergence-base", s.KLBasePath, "-f", s.DatasetPath)
	if s.Chunks > 0 {
		args = append(args, "--chunks", strconv.Itoa(s.Chunks))
	}
	return args, nil
}

// baseArgs validates the spec's shared fields and assembles the
// leading argument list: model, the pre-mapped config flags, and the
// fixed evaluation context.
func (s Spec) baseArgs() ([]string, error) {
	if s.Binary == "" {
		return nil, fmt.Errorf("evaluation spec has no binary path")
	}
	if s.ModelPath == "" {
		return nil, fmt.Errorf("evaluation spec has no model path")
	}
	if s.DatasetPath == "" {
		return nil, fmt.Errorf("evaluation spec has no dataset path")
	}
	args := []string{"--model", s.ModelPath}
	args = append(args, s.Flags...)
	args = append(args, "--ctx-size", strconv.Itoa(EvalContextSize))
	return args, nil
}

// tailLimit bounds the combined output kept for parsing and error
// tails: the last 64 KB.
const tailLimit = 64 * 1024

// errorTailBytes bounds the slice of the combined tail embedded in a
// nonzero-exit error message.
const errorTailBytes = 4 * 1024

// tailBuffer is a bounded ring that keeps only the last limit bytes of
// everything written to it. It is the single destination for the
// process's stdout AND stderr: llama.cpp's logger routes plain LOG
// lines to stdout but LOG_INF to stderr (common/log.cpp:89-92), and
// the final Perplexity and Winogrande score lines are LOG_INF — a
// stdout-only parser would miss two of the four modes. The buffer is
// both the parse input and the error-tail source.
type tailBuffer struct {
	mu    sync.Mutex
	buf   []byte
	limit int
}

func newTailBuffer(limit int) *tailBuffer {
	return &tailBuffer{limit: limit}
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > t.limit {
		t.buf = t.buf[len(t.buf)-t.limit:]
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}

// runProcess executes binary with args under ctx and returns the tail
// of its combined stdout+stderr.
//
// The binary's directory is prepended to LD_LIBRARY_PATH so co-located
// libllama.so / libggml*.so resolve — the way the codebase already
// launches build binaries (internal/api/jobs_env.go:72-87 and
// process.Manager). Without it every eval fails to start.
//
// exec.CommandContext kills the process when ctx is cancelled; the
// error is wrapped around ctx.Err() so callers can identify
// cancellation with errors.Is. A nonzero exit returns an error
// carrying the tail of the combined output.
func runProcess(ctx context.Context, binary string, args []string) (string, error) {
	cmd := exec.CommandContext(ctx, binary, args...)

	binDir := filepath.Dir(binary)
	env := os.Environ()
	prepended := false
	for i, kv := range env {
		if strings.HasPrefix(kv, "LD_LIBRARY_PATH=") {
			env[i] = "LD_LIBRARY_PATH=" + binDir + string(os.PathListSeparator) + kv[len("LD_LIBRARY_PATH="):]
			prepended = true
			break
		}
	}
	if !prepended {
		env = append(env, "LD_LIBRARY_PATH="+binDir)
	}
	cmd.Env = env

	combined := newTailBuffer(tailLimit)
	cmd.Stdout = combined
	cmd.Stderr = combined

	err := cmd.Run()
	if err == nil {
		return combined.String(), nil
	}
	if ctx.Err() != nil {
		return combined.String(), fmt.Errorf("evaluation cancelled: %w", ctx.Err())
	}
	return combined.String(), fmt.Errorf("%s failed: %v — output tail: %s", filepath.Base(binary), err, tailSnippet(combined.String()))
}

// tailSnippet returns the last errorTailBytes of s, for embedding in
// error messages.
func tailSnippet(s string) string {
	if len(s) <= errorTailBytes {
		return s
	}
	return "…" + s[len(s)-errorTailBytes:]
}

// Run executes one evaluation and parses the tool's combined output
// into a Result.
func Run(ctx context.Context, spec Spec) (Result, error) {
	args, err := spec.args()
	if err != nil {
		return Result{}, err
	}
	out, err := runProcess(ctx, spec.Binary, args)
	if err != nil {
		return Result{}, err
	}
	return parse(spec, out)
}

// GenerateKLBase runs the base-logits GENERATION for a KL comparison:
// spec.KLBasePath is the output destination, spec.Chunks the optional
// chunk cap. It shares Run's execution model (combined buffer, after
// exit) but returns only the error — there is no score to parse (see
// klBaseArgs). The base file is written by the tool itself; the
// caller (phase 02/03) owns the .partial rename discipline and the
// progress polling.
func GenerateKLBase(ctx context.Context, spec Spec) error {
	args, err := spec.klBaseArgs()
	if err != nil {
		return err
	}
	_, err = runProcess(ctx, spec.Binary, args)
	return err
}
