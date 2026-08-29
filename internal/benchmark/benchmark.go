package benchmark

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/evaluate"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

const (
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
)

// BenchmarkRun is one complete benchmark execution.
type BenchmarkRun struct {
	ID        string    `json:"id"`
	JobID     string    `json:"job_id,omitempty"` // owning job; "adhoc" for migrated/legacy and quick-bench runs
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"`
	Error     string    `json:"error,omitempty"`

	// What was tested
	ModelID   string  `json:"model_id"`
	ModelName string  `json:"model_name"`
	Quant     string  `json:"quant"`
	SizeGiB   float64 `json:"size_gib"` // model file size, bytes/1024³

	// LegacySizeGB reads files written before the size_gib rename (the
	// value was always binary GiB despite the name); load() folds it
	// into SizeGiB and clears it so it never persists again.
	LegacySizeGB float64 `json:"size_gb,omitempty"`

	// Configuration snapshot
	Config ConfigSnapshot `json:"config"`

	// Build info. The flat BuildID/BuildRef/BuildProfile fields predate
	// the Build snapshot and are preserved for already-persisted runs;
	// new runs populate Build with the full snapshot. Use EffectiveBuild()
	// to read; it falls back to the flat fields for legacy data.
	BuildID      string        `json:"build_id"`
	BuildRef     string        `json:"build_ref"`
	BuildProfile string        `json:"build_profile"`
	Build        BuildSnapshot `json:"build,omitempty"`

	// Hardware
	GPUs []GPUSnapshot `json:"gpus"`

	// Parameters
	Preset       string `json:"preset"`
	PromptTokens []int  `json:"prompt_tokens"`
	GenTokens    int    `json:"gen_tokens"`

	// Results
	Results     []BenchmarkResult   `json:"results,omitempty"`
	Summary     *BenchmarkSummary   `json:"summary,omitempty"`
	LlamaBench  *LlamaBenchResult   `json:"llama_bench,omitempty"`
	LlamaBenchy []LlamaBenchyResult `json:"llama_benchy,omitempty"`

	// Eval holds capability-evaluation scores (perplexity, KL
	// divergence, HellaSwag, Winogrande) for runs of capability
	// presets. Nil on performance runs.
	Eval *EvalScores `json:"eval,omitempty"`

	// Command line that was actually executed for benchy presets, captured
	// at run time so the detail view and "About" modal can disclose it.
	BenchyCommand string `json:"benchy_command,omitempty"`

	// Warnings (non-fatal issues during the run)
	Warnings []string `json:"warnings,omitempty"`

	// Progress (transient, not persisted — only meaningful while running)
	ProgressDetail string `json:"progress_detail,omitempty"`

	// Duration
	DurationMs int64 `json:"duration_ms,omitempty"`

	// SweepValues records which point of a parameter sweep this run
	// measured, copied from its cell. Stored on the run so results stay
	// self-describing when compared across jobs, or after a job is
	// deleted and its runs are orphaned to Ad-Hoc.
	SweepValues map[string]string `json:"sweep_values,omitempty"`

	// ConfigUnverified marks a run whose recorded Config may not reflect
	// what llama-server actually ran. Set by the v2→v3 migration on runs
	// from jobs that declared overrides back when overrides were never
	// applied. Never set on runs produced after that fix.
	ConfigUnverified bool `json:"config_unverified,omitempty"`
}

// ConfigSnapshot freezes model config at benchmark time.
type ConfigSnapshot struct {
	GPULayers      int    `json:"gpu_layers"`
	ContextSize    int    `json:"context_size"`
	GPUAssign      string `json:"gpu_assign,omitempty"`
	TensorSplit    string `json:"tensor_split,omitempty"`
	FlashAttention bool   `json:"flash_attention"`
	KVCacheQuant   string `json:"kv_cache_quant,omitempty"`
	DirectIO       bool   `json:"direct_io,omitempty"`
	Threads        int    `json:"threads"`
	BatchSize      int    `json:"batch_size,omitempty"`
	UBatchSize     int    `json:"ubatch_size,omitempty"`
	SpecType       string `json:"spec_type,omitempty"`
	DraftModelPath string `json:"draft_model_path,omitempty"`
	DraftMax       int    `json:"draft_max,omitempty"`
	DraftMin       int    `json:"draft_min,omitempty"`
	DraftPMin      string `json:"draft_p_min,omitempty"`
	NgramSizeN     int    `json:"ngram_size_n,omitempty"`
	NgramSizeM     int    `json:"ngram_size_m,omitempty"`

	// PLEMode and ExtraFlags reach llama-server through the preset INI
	// (tensor-read-lazy, and the raw flag text appended verbatim). Both
	// are load-time settings that never reach the capability evaluation
	// — see the excluded list on evaluate.MapConfigFlags.
	PLEMode    string `json:"ple_mode,omitempty"`
	ExtraFlags string `json:"extra_flags,omitempty"`
}

// GPUSnapshot captures GPU hardware at benchmark time.
type GPUSnapshot struct {
	Index        int    `json:"index"`
	Name         string `json:"name"`
	VRAMTotalMiB int    `json:"vram_total_mib"` // as reported by nvidia-smi / sysfs, binary MiB

	// LegacyVRAMTotalMB reads files written before the vram_total_mib
	// rename; load() folds it into VRAMTotalMiB and clears it.
	LegacyVRAMTotalMB int `json:"vram_total_mb,omitempty"`
}

// BuildSnapshot freezes the llama.cpp build that produced a benchmark
// run. Captured at run start so deleting/rebuilding the build later
// doesn't strand the result without context.
type BuildSnapshot struct {
	ID         string            `json:"id"`
	Tag        string            `json:"tag,omitempty"`
	Profile    string            `json:"profile"` // rocm | cuda | vulkan | metal | cpu
	Vendor     string            `json:"vendor"`  // currently == Profile; reserved for future split
	GitSHA     string            `json:"git_sha,omitempty"`
	GitRef     string            `json:"git_ref,omitempty"`
	CMakeFlags map[string]string `json:"cmake_flags,omitempty"`
	BinaryPath string            `json:"binary_path,omitempty"`
}

// BenchmarkResult is one test point.
type BenchmarkResult struct {
	PromptTokens    int     `json:"prompt_tokens"`
	GenTokens       int     `json:"gen_tokens"`
	Repetition      int     `json:"repetition"`
	PromptTokPerSec float64 `json:"prompt_tok_per_sec"`
	GenTokPerSec    float64 `json:"gen_tok_per_sec"`
	TTFTMs          float64 `json:"ttft_ms"`
	TotalMs         float64 `json:"total_ms"`
}

// BenchmarkSummary holds aggregated stats.
type BenchmarkSummary struct {
	// AvgPromptTokPerSec and AvgGenTokPerSec average across every result
	// in the run, including different prompt sizes. Retained for stored
	// data and backward compatibility, but they are not comparable to
	// llama-bench or to runs using a different preset — use PerSize.
	AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
	AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
	AvgTTFTMs          float64 `json:"avg_ttft_ms"`
	MinGenTokPerSec    float64 `json:"min_gen_tok_per_sec"`
	MaxGenTokPerSec    float64 `json:"max_gen_tok_per_sec"`

	// PerSize reports one figure per fixed prompt length, with standard
	// deviation across repetitions — the shape llama-bench uses and the
	// only form comparable to externally published numbers.
	PerSize []SizeSummary `json:"per_size,omitempty"`
}

// SizeSummary is one prompt length's results, equivalent to a single
// llama-bench row (pp512, pp2048, …).
type SizeSummary struct {
	PromptTokens int     `json:"prompt_tokens"`
	GenTokens    int     `json:"gen_tokens"`
	Count        int     `json:"count"`
	PPMean       float64 `json:"pp_mean"`
	PPStd        float64 `json:"pp_std"`
	TGMean       float64 `json:"tg_mean"`
	TGStd        float64 `json:"tg_std"`
	AvgTTFTMs    float64 `json:"avg_ttft_ms"`
}

// Label renders the llama-bench-style name for this row, e.g. "pp6310".
func (s SizeSummary) Label() string { return fmt.Sprintf("pp%d", s.PromptTokens) }

// SizeRows returns the per-prompt-size rows for display, one per fixed
// prompt length as llama-bench reports them.
//
// Falls back to a single row carrying the run's mixed average when there
// is no per-size data — a legacy run, or one that failed before
// producing results. That row has PromptTokens 0, which the templates
// render as "mixed" rather than as a comparable figure.
func (r BenchmarkRun) SizeRows() []SizeSummary {
	if r.Summary == nil {
		return nil
	}
	if len(r.Summary.PerSize) > 0 {
		return r.Summary.PerSize
	}
	return []SizeSummary{{
		PromptTokens: 0,
		Count:        len(r.Results),
		PPMean:       r.Summary.AvgPromptTokPerSec,
		TGMean:       r.Summary.AvgGenTokPerSec,
		AvgTTFTMs:    r.Summary.AvgTTFTMs,
	}}
}

// LlamaBenchResult holds raw inference benchmark data.
type LlamaBenchResult struct {
	PromptTokPerSec float64 `json:"pp_avg_ts"`
	GenTokPerSec    float64 `json:"tg_avg_ts"`
	PromptTokens    int     `json:"pp_tokens"`
	GenTokens       int     `json:"tg_tokens"`
	Repetitions     int     `json:"repetitions"`
}

// EvalScores holds the capability-evaluation result of one benchmark
// run: nil on performance-only runs, and each field group omits itself
// when its mode did not produce it.
//
// It is an ALIAS for evaluate.Result, not a copy of it. The engine
// cannot import this package (it must stay free of the job/router
// world), so the two were previously the same seventeen fields declared
// twice and kept in step by hand — a schema addition made in one and
// forgotten in the other loses data silently, with nothing to fail. An
// alias makes them one type: the runner assigns the engine's result
// directly, and there is no copy to drift.
type EvalScores = evaluate.Result

// TimingSample is one observed timing from real usage.
type TimingSample struct {
	Timestamp       time.Time `json:"ts"`
	ModelID         string    `json:"model"`
	PromptTokens    int       `json:"prompt_n"`
	GenTokens       int       `json:"gen_n"`
	PromptTokPerSec float64   `json:"prompt_tps"`
	GenTokPerSec    float64   `json:"gen_tps"`
}

// PresetSourceInternal drives the in-process API benchmark loop in
// runner.go (real chat completions through the router). PresetSourceBenchy
// shells out to `uvx llama-benchy` against the same router. Empty defaults
// to internal so older preset definitions stay valid.
const (
	PresetSourceInternal   = "internal"
	PresetSourceBenchy     = "benchy"
	PresetSourceCapability = "capability"
)

// Preset defines benchmark parameters.
type Preset struct {
	Name         string
	Label        string
	Description  string
	Source       string // "" | "internal" | "benchy" | "capability"
	PromptTokens []int
	GenTokens    int
	Repetitions  int
	Concurrency  []int // benchy only; defaults to [1] if empty

	// Capability presets only (Source == PresetSourceCapability).
	// EvalMode names the evaluation the cell runs; EvalTasks and
	// EvalChunks are the run limits (0 = full run). Performance presets
	// leave all three zero.
	EvalMode   evaluate.Mode
	EvalTasks  int
	EvalChunks int
}

// EffectiveSource returns the dispatch key, defaulting empty → internal.
func (p Preset) EffectiveSource() string {
	if p.Source == "" {
		return PresetSourceInternal
	}
	return p.Source
}

// Presets returns the available benchmark presets.
func Presets() []Preset {
	return []Preset{
		{
			Name:         "internal-quick",
			Label:        "internal-quick — 1 rep, 256-token prompt (~10s)",
			Description:  "Single end-to-end request with a 256-token prompt and 128 generated tokens. Sanity check that the model loads and runs.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{256}, GenTokens: 128, Repetitions: 1,
		},
		{
			Name:         "internal-standard",
			Label:        "internal-standard — 3 reps × 3 prompt sizes (~1 min)",
			Description:  "Three repetitions of end-to-end requests at 128, 512, and 2048-token prompts (128 gen tokens each).",
			Source:       PresetSourceInternal,
			PromptTokens: []int{128, 512, 2048}, GenTokens: 128, Repetitions: 3,
		},
		{
			Name:         "internal-thorough",
			Label:        "internal-thorough — 5 reps × 4 prompt sizes up to 8K (~5 min)",
			Description:  "Five repetitions at 128 / 512 / 2048 / 8192-token prompts with 256 generated tokens each. Stresses long-context performance.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{128, 512, 2048, 8192}, GenTokens: 256, Repetitions: 5,
		},
		{
			Name:         "internal-long-ctx",
			Label:        "internal-long-ctx — 1 rep, 32K prompt / 512 gen",
			Description:  "Single 32768-token prompt with 512 generated tokens. Stresses KV cache, flash-attention, and KV quantization on a long context.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{32768}, GenTokens: 512, Repetitions: 1,
		},
		{
			Name:         "internal-long-ctx-thorough",
			Label:        "internal-long-ctx-thorough — 3 reps × 4 prompt sizes 8K–64K (~20 min)",
			Description:  "Three repetitions at 8192 / 16384 / 32768 / 65536-token prompts with 512 generated tokens each. Long-context companion to internal-long-ctx: enough repetitions to see run-to-run variance, and enough prompt sizes to show how prefill throughput scales with context depth. Prompt sizes are nominal — prompts are built at an estimated 4 chars/token, so the tokenized length typically comes out 10-25% lower depending on the tokenizer (check prompt_tokens in the results). Needs a model context of at least 64K.",
			Source:       PresetSourceInternal,
			PromptTokens: []int{8192, 16384, 32768, 65536}, GenTokens: 512, Repetitions: 3,
		},
		{
			Name:         "benchy-quick",
			Label:        "benchy-quick — 1 rep, 512 prompt / 32 gen via llama-benchy (~10s)",
			Description:  "Single-shot llama-benchy run against the router. Smoke test for the API path; works with sharded GGUFs.",
			Source:       PresetSourceBenchy,
			PromptTokens: []int{512}, GenTokens: 32, Repetitions: 1, Concurrency: []int{1},
		},
		{
			Name:         "benchy-standard",
			Label:        "benchy-standard — 3 reps, 2048 prompt / 128 gen via llama-benchy (~1 min)",
			Description:  "Three-run llama-benchy benchmark at 2048-token prompts. Replaces the legacy llama-bench raw inference test for sharded models.",
			Source:       PresetSourceBenchy,
			PromptTokens: []int{2048}, GenTokens: 128, Repetitions: 3, Concurrency: []int{1},
		},
		// Capability presets run llama-perplexity directly against the
		// model (no router), all at the fixed evaluation context of
		// 512 tokens — the perplexity chunk size, and the context
		// llama.cpp's published wikitext figures use. That fixed context
		// is what makes "all chunks" / "all tasks" a defined quantity
		// for the full variants.
		{
			Name:        "perplexity-quick",
			Label:       "perplexity-quick — 100 chunks of wikitext-2 (~2-5 min/model)",
			Description: "Perplexity over the first 100 chunks of the wikitext-2 test set at a fixed 512-token context: lower is better, and differences between quants of one model show up in the error bar. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModePerplexity,
			EvalChunks:  100,
		},
		{
			Name:        "perplexity-full",
			Label:       "perplexity-full — all ~650 chunks of wikitext-2 (~15-40 min/model)",
			Description: "Perplexity over the entire wikitext-2 test set (about 650 chunks at the fixed 512-token context). The publishable number; comparable to the figures llama.cpp quotes for models. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModePerplexity,
			EvalChunks:  0,
		},
		{
			Name:        "kl-divergence-quick",
			Label:       "kl-divergence-quick — 100 chunks vs the reference quant (~2-5 min/model)",
			Description: "KL divergence between this model's logits and the reference model's over the first 100 chunks of wikitext-2 at the fixed 512-token context: zero means identical probabilities, so it measures how much a quantization changed the model. The reference defaults to the largest installed quant of the same repo; its logits are generated once (a visible step) and cached. The inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeKLDiv,
			EvalChunks:  100,
		},
		{
			Name:        "kl-divergence-full",
			Label:       "kl-divergence-full — all ~650 chunks vs the reference quant (~15-40 min/model; base is tens of GiB for large-vocab models)",
			Description: "KL divergence against the reference model over the entire wikitext-2 test set at the fixed 512-token context. The full-run reference base is tens of GiB for large-vocab models and is gated by a disk-space check before generation; it is generated once and cached. The inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeKLDiv,
			EvalChunks:  0,
		},
		{
			Name:        "hellaswag-quick",
			Label:       "hellaswag-quick — 400 HellaSwag tasks (~1-5 min/model)",
			Description: "HellaSwag commonsense accuracy on 400 tasks: the percentage of candidate endings the model ranks first, matching the preprocessing llama.cpp uses so scores are comparable to published numbers. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeHellaSwag,
			EvalTasks:   400,
		},
		{
			Name:        "hellaswag-full",
			Label:       "hellaswag-full — all ~10K HellaSwag tasks (~1-3 h/model)",
			Description: "HellaSwag commonsense accuracy on the full validation set (~10K tasks), the publishable number. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeHellaSwag,
			EvalTasks:   0,
		},
		{
			Name:        "winogrande-quick",
			Label:       "winogrande-quick — 400 Winogrande tasks (~1-3 min/model)",
			Description: "Winogrande pronoun-resolution accuracy on 400 tasks of the debiased eval set: the percentage of sentences the model completes with the correct referent. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeWinogrande,
			EvalTasks:   400,
		},
		{
			Name:        "winogrande-full",
			Label:       "winogrande-full — all ~1.2K Winogrande tasks (~5-15 min/model)",
			Description: "Winogrande pronoun-resolution accuracy on the entire debiased eval set (~1.2K tasks), the publishable number. Runs llama-perplexity directly, so the inference server is offline while the cell runs.",
			Source:      PresetSourceCapability,
			EvalMode:    evaluate.ModeWinogrande,
			EvalTasks:   0,
		},
	}
}

// presetAliases maps the pre-rename preset names (used by data persisted
// before the internal-* / benchy-* split) onto their current equivalents,
// so old runs render and re-runs find the right preset.
var presetAliases = map[string]string{
	"quick":    "internal-quick",
	"standard": "internal-standard",
	"thorough": "internal-thorough",
}

// GetPreset returns a preset by name, falling back to "internal-standard".
func GetPreset(name string) Preset {
	if alias, ok := presetAliases[name]; ok {
		name = alias
	}
	for _, p := range Presets() {
		if p.Name == name {
			return p
		}
	}
	return Presets()[1] // internal-standard
}

// EffectiveBuild returns the run's Build snapshot, falling back to a
// minimal snapshot synthesized from the legacy flat fields when the run
// was persisted before BuildSnapshot existed.
func (r *BenchmarkRun) EffectiveBuild() BuildSnapshot {
	if r.Build.ID != "" || r.Build.GitRef != "" {
		return r.Build
	}
	return BuildSnapshot{
		ID:      r.BuildID,
		GitRef:  r.BuildRef,
		Profile: r.BuildProfile,
		Vendor:  r.BuildProfile,
	}
}

// GPUSnapshotsFromMetrics converts monitor metrics to GPU snapshots.
func GPUSnapshotsFromMetrics(m monitor.Metrics) []GPUSnapshot {
	snaps := make([]GPUSnapshot, len(m.GPU))
	for i, g := range m.GPU {
		snaps[i] = GPUSnapshot{
			Index:        g.Index,
			Name:         g.Name,
			VRAMTotalMiB: g.VRAMTotalMB,
		}
	}
	return snaps
}

// BuildResolver resolves a build ID to its full snapshot. Implemented by
// the api layer over *builder.Builder; passed in so the benchmark
// package doesn't import builder directly.
//
// A nil resolver, or one that returns the zero value, signals "not
// known" — the migration falls back to the legacy flat fields.
type BuildResolver func(buildID string) BuildSnapshot

// Store manages benchmark persistence and timing samples.
type Store struct {
	mu       sync.RWMutex
	dataDir  string
	runs     []BenchmarkRun
	jobs     []BenchmarkJob
	resolver BuildResolver

	timingsMu sync.RWMutex
	timings   map[string][]TimingSample // model ID → ring buffer
}

const maxTimingSamples = 1000

// schemaVersion is the on-disk envelope version this build writes. v1
// was a bare JSON array of runs; v2 wraps them with a jobs list; v3
// flags runs whose recorded config was never actually applied; v4
// renames size_gb → size_gib and vram_total_mb → vram_total_mib (the
// values were always binary units — the old names were wrong).
const schemaVersion = 4

// benchmarkFile is the v2 envelope. v1 files are detected by an
// unmarshal failure into this shape and a successful retry as []BenchmarkRun.
type benchmarkFile struct {
	Version int            `json:"version"`
	Jobs    []BenchmarkJob `json:"jobs"`
	Runs    []BenchmarkRun `json:"runs"`
}

// markUnverifiedConfigs is the v2→v3 migration. Before v3, a job's
// ConfigOverrides were merged into each run's recorded ConfigSnapshot
// but never applied to llama-server — the cell benchmarked the model's
// saved config while reporting the overridden one. The throughput
// numbers are real; the config they are attributed to is not.
//
// We cannot recover what actually ran, so flag every run belonging to a
// job that declared overrides and let the UI say so.
//
// Known gap: runs whose job was deleted before this upgrade were
// re-parented to the synthetic Ad-Hoc job, which has no Overrides, so
// they cannot be identified here and stay unflagged. They are also
// indistinguishable from genuine ad-hoc runs, whose recorded config was
// always accurate — flagging the whole Ad-Hoc job would mislabel those
// instead. Affected runs are limited to pre-v3 override jobs the user
// deleted while keeping results.
func markUnverifiedConfigs(jobs []BenchmarkJob, runs []BenchmarkRun) bool {
	overridden := make(map[string]bool, len(jobs))
	for _, j := range jobs {
		if j.Overrides != nil {
			overridden[j.ID] = true
		}
	}
	if len(overridden) == 0 {
		return false
	}
	changed := false
	for i := range runs {
		if overridden[runs[i].JobID] && !runs[i].ConfigUnverified {
			runs[i].ConfigUnverified = true
			changed = true
		}
	}
	return changed
}

// NewStore creates a store and loads persisted benchmarks. resolver may
// be nil; if provided, the v1→v2 migration uses it to backfill Build
// snapshots for runs whose build still exists in the builder.
func NewStore(dataDir string, resolver BuildResolver) *Store {
	s := &Store{
		dataDir:  dataDir,
		resolver: resolver,
		timings:  make(map[string][]TimingSample),
	}
	s.load()
	return s
}

// List returns all benchmark runs, newest first.
func (s *Store) List() []BenchmarkRun {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]BenchmarkRun, len(s.runs))
	copy(out, s.runs)
	sort.Slice(out, func(i, j int) bool {
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

// Get returns a benchmark run by ID.
func (s *Store) Get(id string) (*BenchmarkRun, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range s.runs {
		if s.runs[i].ID == id {
			run := s.runs[i]
			return &run, nil
		}
	}
	return nil, fmt.Errorf("benchmark not found: %s", id)
}

// Save adds or updates a benchmark run. Runs with no JobID are assigned
// to the synthetic Ad-Hoc Runs job so the "every run belongs to a job"
// invariant holds across the existing single-run code path.
func (s *Store) Save(run BenchmarkRun) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run.JobID == "" {
		run.JobID = AdhocJobID
		if !s.hasJobLocked(AdhocJobID) {
			s.jobs = append(s.jobs, newAdhocJob(run.CreatedAt))
		}
	}
	for i := range s.runs {
		if s.runs[i].ID == run.ID {
			s.runs[i] = run
			s.persist()
			return
		}
	}
	s.runs = append(s.runs, run)
	s.persist()
}

// Delete removes a benchmark run.
func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.runs {
		if s.runs[i].ID == id {
			s.runs = append(s.runs[:i], s.runs[i+1:]...)
			s.persist()
			return nil
		}
	}
	return fmt.Errorf("benchmark not found: %s", id)
}

// ListJobs returns all jobs, newest CreatedAt first. The synthetic
// AdhocJobID always sorts last, regardless of timestamp, so the user's
// real batch jobs surface above their history.
func (s *Store) ListJobs() []BenchmarkJob {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]BenchmarkJob, len(s.jobs))
	copy(out, s.jobs)
	sort.SliceStable(out, func(i, j int) bool {
		if (out[i].ID == AdhocJobID) != (out[j].ID == AdhocJobID) {
			return out[j].ID == AdhocJobID
		}
		return out[i].CreatedAt.After(out[j].CreatedAt)
	})
	return out
}

// GetJob returns a job by ID.
func (s *Store) GetJob(id string) (*BenchmarkJob, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			job := s.jobs[i]
			return &job, nil
		}
	}
	return nil, fmt.Errorf("job not found: %s", id)
}

// SaveJob adds or updates a job.
func (s *Store) SaveJob(job BenchmarkJob) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := range s.jobs {
		if s.jobs[i].ID == job.ID {
			s.jobs[i] = job
			s.persist()
			return
		}
	}
	s.jobs = append(s.jobs, job)
	s.persist()
}

// DeleteJob removes a job. The disposition controls what happens to its
// runs: DeleteCascade removes them with the job; DeleteOrphan reassigns
// them to the AdhocJobID. Deleting AdhocJobID itself is rejected — it's
// the migration target and the home of the existing single-run path.
func (s *Store) DeleteJob(id string, disposition DeleteDisposition) error {
	if id == AdhocJobID {
		return fmt.Errorf("cannot delete the synthetic %q job", AdhocJobID)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	idx := -1
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			idx = i
			break
		}
	}
	if idx == -1 {
		return fmt.Errorf("job not found: %s", id)
	}
	switch disposition {
	case DeleteCascade:
		filtered := s.runs[:0]
		for _, r := range s.runs {
			if r.JobID != id {
				filtered = append(filtered, r)
			}
		}
		s.runs = filtered
	case DeleteOrphan:
		if !s.hasJobLocked(AdhocJobID) {
			s.jobs = append(s.jobs, newAdhocJob(time.Now()))
		}
		for i := range s.runs {
			if s.runs[i].JobID == id {
				s.runs[i].JobID = AdhocJobID
			}
		}
	default:
		return fmt.Errorf("unknown disposition: %q (want %q or %q)", disposition, DeleteCascade, DeleteOrphan)
	}
	s.jobs = append(s.jobs[:idx], s.jobs[idx+1:]...)
	s.persist()
	return nil
}

// UpdateJobDefinition replaces a job's editable fields (name/description/
// matrix/overrides) and rebuilds its cell list. Cells whose (model, build,
// preset) coordinate still exists in the new matrix AND were already
// completed are carried over with their run intact; everything else
// starts fresh as pending. Runs no longer reachable from any cell are
// orphaned to the Ad-Hoc job so the user keeps the history. The caller
// must Submit the returned job to actually re-run it.
//
// Refuses to edit the synthetic adhoc job. Does not check whether the
// queue is busy — that's the JobQueue's responsibility on Submit.
// JobDefinition is the editable part of a job. Grouped into a struct
// because the positional form had grown to seven parameters and adding
// sweeps would have made eight.
type JobDefinition struct {
	Name        string
	Description string
	ModelIDs    []string
	BuildIDs    []string
	Presets     []string
	Overrides   *ConfigOverrides
	Sweeps      []SweepAxis
	KLReference string
}

// cellIdentity keys a cell for match-up across an edit. Sweep values are
// part of the identity: without them, editing a swept job would match a
// completed cell to a different sweep point and report its result under
// the wrong configuration.
type cellIdentity struct {
	Model, Build, Preset string
	Sweep                string
	Overrides            string
}

// overrideKey renders a job's fixed overrides into the cell identity.
// Without it, editing a single-value parameter (which lands in Overrides
// rather than Sweeps) leaves every completed cell matching, so results
// measured at the old value are carried forward and re-attributed to the
// new one. Harmless while overrides did nothing; not anymore.
func overrideKey(o *ConfigOverrides) string {
	if o == nil {
		return ""
	}
	b, err := json.Marshal(o)
	if err != nil {
		return ""
	}
	// Every field is omitempty, so an all-nil struct marshals to "{}".
	// That must key the same as nil: they mean the same thing, and
	// differing would drop every completed cell to pending and re-parent
	// its runs to Ad-Hoc on an unrelated edit.
	if string(b) == "{}" {
		return ""
	}
	return string(b)
}

func identify(c JobCell) cellIdentity {
	names := make([]string, 0, len(c.SweepValues))
	for k := range c.SweepValues {
		names = append(names, k)
	}
	sort.Strings(names)
	var b strings.Builder
	for _, n := range names {
		fmt.Fprintf(&b, "%s=%s;", n, c.SweepValues[n])
	}
	return cellIdentity{Model: c.ModelID, Build: c.BuildID, Preset: c.Preset, Sweep: b.String()}
}

// identifyIn keys a cell within a job, folding in that job's fixed
// overrides so a changed override invalidates prior results.
func identifyIn(c JobCell, o *ConfigOverrides) cellIdentity {
	id := identify(c)
	id.Overrides = overrideKey(o)
	return id
}

func (s *Store) UpdateJobDefinition(id string, def JobDefinition) (*BenchmarkJob, error) {
	name, description := def.Name, def.Description
	modelIDs, buildIDs, presets := def.ModelIDs, def.BuildIDs, def.Presets
	overrides := def.Overrides
	klReference := def.KLReference
	if id == AdhocJobID {
		return nil, fmt.Errorf("cannot edit the synthetic %q job", AdhocJobID)
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := -1
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, fmt.Errorf("job not found: %s", id)
	}
	job := s.jobs[idx]

	prev := make(map[cellIdentity]JobCell, len(job.Cells))
	for _, c := range job.Cells {
		prev[identifyIn(c, job.Overrides)] = c
	}

	newCells := ExpandCellsWithSweeps(modelIDs, buildIDs, presets, def.Sweeps)
	keptRuns := make(map[string]bool)
	for i := range newCells {
		k := identifyIn(newCells[i], overrides)
		if old, ok := prev[k]; ok && old.Status == CellStatusCompleted {
			newCells[i] = old
			if old.BenchmarkRunID != "" {
				keptRuns[old.BenchmarkRunID] = true
			}
		}
	}

	// Anything previously linked to this job that didn't survive into a
	// kept cell becomes adhoc, preserving the run history.
	orphaned := false
	for i := range s.runs {
		if s.runs[i].JobID != id || keptRuns[s.runs[i].ID] {
			continue
		}
		s.runs[i].JobID = AdhocJobID
		orphaned = true
	}
	if orphaned && !s.hasJobLocked(AdhocJobID) {
		s.jobs = append(s.jobs, newAdhocJob(time.Now()))
		// hasJobLocked failed because the slice didn't contain an adhoc
		// entry; appending it can move the backing array, so re-find the
		// index of the job we're editing.
		for i := range s.jobs {
			if s.jobs[i].ID == id {
				idx = i
				break
			}
		}
	}

	job.Name = name
	job.Description = description
	job.ModelIDs = modelIDs
	job.BuildIDs = buildIDs
	job.Presets = presets
	job.Overrides = overrides
	job.Sweeps = def.Sweeps
	job.KLReference = klReference
	job.Cells = newCells
	job.Status = JobStatusPending
	job.StartedAt = time.Time{}
	job.FinishedAt = time.Time{}
	s.jobs[idx] = job

	s.persist()
	out := job
	return &out, nil
}

// RunsForJob returns all runs belonging to the given job, newest first.
func (s *Store) RunsForJob(jobID string) []BenchmarkRun {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var out []BenchmarkRun
	for _, r := range s.runs {
		if r.JobID == jobID {
			out = append(out, r)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].CreatedAt.After(out[j].CreatedAt) })
	return out
}

// RecordTiming adds a passive timing sample.
func (s *Store) RecordTiming(sample TimingSample) {
	s.timingsMu.Lock()
	defer s.timingsMu.Unlock()
	samples := s.timings[sample.ModelID]
	samples = append(samples, sample)
	if len(samples) > maxTimingSamples {
		samples = samples[len(samples)-maxTimingSamples:]
	}
	s.timings[sample.ModelID] = samples
}

// Timings returns recent timing samples for a model (or all models if empty).
func (s *Store) Timings(modelID string) []TimingSample {
	s.timingsMu.RLock()
	defer s.timingsMu.RUnlock()
	if modelID != "" {
		out := make([]TimingSample, len(s.timings[modelID]))
		copy(out, s.timings[modelID])
		return out
	}
	var all []TimingSample
	for _, samples := range s.timings {
		all = append(all, samples...)
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].Timestamp.After(all[j].Timestamp)
	})
	return all
}

// TimingSummary returns aggregated timing stats per model.
// promptBuckets groups observed requests by prompt length. Prompt
// processing throughput rises steeply with prompt size — a 72-token
// prefill cannot reach the rate of a 6000-token one on any hardware,
// because fixed per-request overhead dominates — so a single headline
// number moves whenever the traffic mix moves, independently of how
// fast the server actually is.
var promptBuckets = []struct {
	Label string
	Min   int // inclusive
	Max   int // exclusive; 0 = no upper bound
}{
	{"<512", 0, 512},
	{"512–2k", 512, 2048},
	{"2k–8k", 2048, 8192},
	{"8k+", 8192, 0},
}

// TimingSummary aggregates passive timing samples per model.
//
// Rates are token-weighted (total tokens / total time), not a mean of
// per-request rates. Averaging rates gave a 72-token request the same
// weight as a 6000-token one, so the figure tracked how many short
// requests happened to be in the window rather than how fast the server
// was: an observed window of 14 requests, 11 of them sub-200-token
// cache-hit continuations, averaged 351 t/s while the same server
// measured 2400+ t/s on 6k prompts.
func (s *Store) TimingSummary() []TimingModelSummary {
	s.timingsMu.RLock()
	defer s.timingsMu.RUnlock()

	var out []TimingModelSummary
	for modelID, samples := range s.timings {
		if len(samples) == 0 {
			continue
		}
		sum := newTimingAgg()
		buckets := make([]*timingAgg, len(promptBuckets))
		for i := range buckets {
			buckets[i] = newTimingAgg()
		}

		minTok, maxTok := 0, 0
		for i, t := range samples {
			sum.add(t)
			if b := bucketFor(t.PromptTokens); b >= 0 {
				buckets[b].add(t)
			}
			if i == 0 || t.PromptTokens < minTok {
				minTok = t.PromptTokens
			}
			if t.PromptTokens > maxTok {
				maxTok = t.PromptTokens
			}
		}

		m := TimingModelSummary{
			ModelID:            modelID,
			Count:              len(samples),
			AvgGenTokPerSec:    sum.genRate(),
			AvgPromptTokPerSec: sum.promptRate(),
			MinPromptTokens:    minTok,
			MaxPromptTokens:    maxTok,
		}
		for i, b := range promptBuckets {
			if buckets[i].count == 0 {
				continue
			}
			m.Buckets = append(m.Buckets, TimingBucket{
				Label:              b.Label,
				Count:              buckets[i].count,
				AvgPromptTokPerSec: buckets[i].promptRate(),
				AvgGenTokPerSec:    buckets[i].genRate(),
			})
		}
		out = append(out, m)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].ModelID < out[j].ModelID
	})
	return out
}

func bucketFor(promptTokens int) int {
	for i, b := range promptBuckets {
		if promptTokens >= b.Min && (b.Max == 0 || promptTokens < b.Max) {
			return i
		}
	}
	return -1
}

// timingAgg accumulates tokens and the time they took, reconstructing
// each sample's duration from its recorded rate. Summing durations and
// tokens separately is what makes the result token-weighted.
type timingAgg struct {
	count                   int
	promptTokens, genTokens float64
	promptSecs, genSecs     float64
}

func newTimingAgg() *timingAgg { return &timingAgg{} }

func (a *timingAgg) add(t TimingSample) {
	a.count++
	if t.PromptTokPerSec > 0 && t.PromptTokens > 0 {
		a.promptTokens += float64(t.PromptTokens)
		a.promptSecs += float64(t.PromptTokens) / t.PromptTokPerSec
	}
	if t.GenTokPerSec > 0 && t.GenTokens > 0 {
		a.genTokens += float64(t.GenTokens)
		a.genSecs += float64(t.GenTokens) / t.GenTokPerSec
	}
}

func (a *timingAgg) promptRate() float64 {
	if a.promptSecs == 0 {
		return 0
	}
	return a.promptTokens / a.promptSecs
}

func (a *timingAgg) genRate() float64 {
	if a.genSecs == 0 {
		return 0
	}
	return a.genTokens / a.genSecs
}

// TimingModelSummary is aggregated timing stats for one model.
type TimingModelSummary struct {
	ModelID            string  `json:"model_id"`
	Count              int     `json:"count"`
	AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
	AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
	// Prompt-length range the rates were observed over, so a headline
	// figure can't be mistaken for a hardware measurement.
	MinPromptTokens int            `json:"min_prompt_tokens"`
	MaxPromptTokens int            `json:"max_prompt_tokens"`
	Buckets         []TimingBucket `json:"buckets,omitempty"`
}

// TimingBucket is the same stats restricted to one prompt-length range,
// which is what makes two periods comparable when the traffic mix moved.
type TimingBucket struct {
	Label              string  `json:"label"`
	Count              int     `json:"count"`
	AvgPromptTokPerSec float64 `json:"avg_prompt_tok_per_sec"`
	AvgGenTokPerSec    float64 `json:"avg_gen_tok_per_sec"`
}

func (s *Store) benchmarkPath() string {
	return filepath.Join(s.dataDir, "config", "benchmarks.json")
}

func (s *Store) load() {
	data, err := os.ReadFile(s.benchmarkPath())
	if err != nil {
		return
	}

	dirty := false

	// Try v2 envelope first; on failure, fall back to v1 bare array.
	var file benchmarkFile
	if jsonErr := json.Unmarshal(data, &file); jsonErr == nil && file.Version >= 2 {
		s.jobs = file.Jobs
		s.runs = file.Runs
	} else {
		var runs []BenchmarkRun
		if v1Err := json.Unmarshal(data, &runs); v1Err != nil {
			slog.Error("failed to load benchmarks (neither v2 envelope nor v1 array)", "v2_error", jsonErr, "v1_error", v1Err)
			return
		}
		s.runs = runs
		dirty = true // forces a v2 rewrite at end of load
	}

	// Backfill per-size summaries for runs stored before they existed.
	// The raw per-result data already carries prompt_tokens, so this is a
	// pure re-aggregation — no measurement is lost or invented.
	for i := range s.runs {
		r := &s.runs[i]
		if r.Summary != nil && len(r.Summary.PerSize) == 0 && len(r.Results) > 0 {
			r.Summary.PerSize = computePerSize(r.Results)
			dirty = true
		}
	}

	// v3→v4: fold the misnamed legacy unit fields into their renamed
	// successors. Unconditional rather than version-gated — it's
	// idempotent, and pre-v2 files carry no version at all.
	for i := range s.runs {
		r := &s.runs[i]
		if r.SizeGiB == 0 && r.LegacySizeGB != 0 {
			r.SizeGiB = r.LegacySizeGB
			r.LegacySizeGB = 0
			dirty = true
		}
		for gi := range r.GPUs {
			g := &r.GPUs[gi]
			if g.VRAMTotalMiB == 0 && g.LegacyVRAMTotalMB != 0 {
				g.VRAMTotalMiB = g.LegacyVRAMTotalMB
				g.LegacyVRAMTotalMB = 0
				dirty = true
			}
		}
	}

	// v2→v3: flag runs whose config snapshot was never applied. Runs
	// written by v3+ are already correct, so scope this to older files.
	if file.Version < 3 {
		if markUnverifiedConfigs(s.jobs, s.runs) {
			dirty = true
		}
	}

	// Any benchmark still marked running at startup belongs to a previous
	// process that died mid-run — surface it as failed so it's deletable.
	for i := range s.runs {
		if s.runs[i].Status == StatusRunning {
			s.runs[i].Status = StatusFailed
			if s.runs[i].Error == "" {
				s.runs[i].Error = "interrupted: server restarted before benchmark finished"
			}
			dirty = true
		}
	}

	// Same fixup for jobs and any cells caught mid-flight.
	for i := range s.jobs {
		if s.jobs[i].Status != JobStatusRunning {
			continue
		}
		s.jobs[i].Status = JobStatusFailed
		s.jobs[i].FinishedAt = time.Now()
		for ci := range s.jobs[i].Cells {
			c := &s.jobs[i].Cells[ci]
			if c.Status == CellStatusRunning {
				c.Status = CellStatusFailed
				if c.Error == "" {
					c.Error = "interrupted: server restarted before job finished"
				}
			}
		}
		dirty = true
	}

	// Backfill JobID="adhoc" on every run that lacks one and ensure the
	// adhoc pseudo-job exists. Track the earliest run's CreatedAt so the
	// synthesized adhoc job sorts naturally with history.
	var earliest time.Time
	needsAdhoc := false
	for i := range s.runs {
		if s.runs[i].JobID == "" {
			s.runs[i].JobID = AdhocJobID
			needsAdhoc = true
			dirty = true
		}
		if s.runs[i].JobID == AdhocJobID {
			if earliest.IsZero() || s.runs[i].CreatedAt.Before(earliest) {
				earliest = s.runs[i].CreatedAt
			}
		}
	}
	if needsAdhoc && !s.hasJobLocked(AdhocJobID) {
		s.jobs = append(s.jobs, newAdhocJob(earliest))
	}

	// Backfill Build snapshot for runs that pre-date step 3. Try the
	// resolver first (gives full CMake flags / tag / SHA when the build
	// still exists), then fall back to the legacy flat fields.
	for i := range s.runs {
		if s.runs[i].Build.ID != "" || s.runs[i].Build.GitRef != "" {
			continue
		}
		if s.resolver != nil && s.runs[i].BuildID != "" {
			if snap := s.resolver(s.runs[i].BuildID); snap.ID != "" {
				s.runs[i].Build = snap
				dirty = true
				continue
			}
		}
		if s.runs[i].BuildID != "" || s.runs[i].BuildRef != "" {
			s.runs[i].Build = BuildSnapshot{
				ID:      s.runs[i].BuildID,
				GitRef:  s.runs[i].BuildRef,
				Profile: s.runs[i].BuildProfile,
				Vendor:  s.runs[i].BuildProfile,
			}
			dirty = true
		}
	}

	if dirty {
		s.persist()
	}
}

func (s *Store) hasJobLocked(id string) bool {
	for i := range s.jobs {
		if s.jobs[i].ID == id {
			return true
		}
	}
	return false
}

func (s *Store) persist() {
	os.MkdirAll(filepath.Dir(s.benchmarkPath()), 0o755)
	file := benchmarkFile{
		Version: schemaVersion,
		Jobs:    s.jobs,
		Runs:    s.runs,
	}
	data, err := json.MarshalIndent(file, "", "  ")
	if err != nil {
		slog.Error("failed to marshal benchmarks", "error", err)
		return
	}
	os.WriteFile(s.benchmarkPath(), data, 0o644)
}
