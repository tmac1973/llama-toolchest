package benchmark

import (
	"fmt"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// AdhocJobID is the synthetic catch-all job that holds runs not produced
// by an explicit batch (legacy migrated runs and the still-existing
// single-run / quick-benchmark path).
const AdhocJobID = "adhoc"

const (
	JobKindBatch = "batch"
	JobKindAdhoc = "ad-hoc"
)

const (
	JobStatusPending   = "pending"
	JobStatusRunning   = "running"
	JobStatusCompleted = "completed"
	JobStatusFailed    = "failed"
	JobStatusCanceled  = "canceled"
)

const (
	CellStatusPending   = "pending"
	CellStatusRunning   = "running"
	CellStatusCompleted = "completed"
	CellStatusFailed    = "failed"
	CellStatusSkipped   = "skipped"
)

// DeleteDisposition controls what happens to a job's runs when the job
// is deleted: cascade removes them, orphan reassigns them to AdhocJobID.
type DeleteDisposition string

const (
	DeleteCascade DeleteDisposition = "cascade"
	DeleteOrphan  DeleteDisposition = "orphan"
)

// BenchmarkJob is a named, persistent batch run sweeping a Cartesian
// matrix of {ModelIDs} × {BuildIDs} × {Presets}. The expanded matrix
// lives in Cells.
type BenchmarkJob struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Kind        string `json:"kind"`   // "batch" | "ad-hoc"
	Status      string `json:"status"` // pending|running|completed|failed|canceled

	CreatedAt  time.Time `json:"created_at"`
	StartedAt  time.Time `json:"started_at,omitempty"`
	FinishedAt time.Time `json:"finished_at,omitempty"`

	// Definition (immutable after first run)
	ModelIDs  []string         `json:"model_ids,omitempty"`
	BuildIDs  []string         `json:"build_ids,omitempty"`
	Presets   []string         `json:"presets,omitempty"`
	Overrides *ConfigOverrides `json:"overrides,omitempty"`

	// BaseProfile measures from a saved profile's settings rather than
	// the model's live config. Autotune starts from a profile the user
	// chose, and must neither depend on the live config nor change it.
	BaseProfile *BaseProfile `json:"base_profile,omitempty"`
	// AutotuneID and AutotuneStage link a job to the autotune run that
	// submitted it, and name the stage it measures.
	AutotuneID    string `json:"autotune_id,omitempty"`
	AutotuneStage string `json:"autotune_stage,omitempty"`

	// Sweeps expand the matrix: every combination of every axis becomes
	// its own cell. Overrides still apply to all of them, so a fixed
	// value acts as the baseline for whatever isn't being swept.
	Sweeps []SweepAxis `json:"sweeps,omitempty"`

	// KLReference names the registry model the kl-divergence cells
	// compare against; empty means automatic (the largest installed
	// quant of each model's own HF repo). Ignored by non-KL cells.
	KLReference string `json:"kl_reference,omitempty"`

	// Starts names, per model ID, the starting points the job measures:
	// "" for the model's current settings, or a saved profile's name.
	// Nil (every job before profile comparison) means current settings
	// only, for every model.
	Starts map[string][]string `json:"starts,omitempty"`
	// Profiles are copies of the saved profiles named in Starts, taken
	// when the job was created or last edited. Cells measure these
	// copies, so changing or deleting a profile later does not change
	// what the job measures or what a retry runs.
	Profiles []JobProfile `json:"profiles,omitempty"`

	// Expanded matrix
	Cells []JobCell `json:"cells,omitempty"`
}

// JobProfile is the copy of one saved profile that a job measures.
type JobProfile struct {
	ModelID string             `json:"model_id"`
	Name    string             `json:"name"`
	Config  models.ModelConfig `json:"config"`
	// SavedAt is when the profile was saved, and CopiedAt when the job
	// took this copy of it.
	SavedAt  time.Time `json:"saved_at"`
	CopiedAt time.Time `json:"copied_at"`
	// BuildID is the build the profile was saved on, for the warning
	// shown when the job runs it on another build.
	BuildID string `json:"build_id,omitempty"`
}

// cellProfile is the saved profile a cell measures from: its own
// profile's copy when it names one, otherwise the job-wide BaseProfile
// (autotune and autoconfigure), otherwise nil for the model's current
// settings. A cell naming a profile the job holds no copy of is an
// error rather than a quiet fall back to the current settings, which
// would record the current settings under the profile's name.
func (j *BenchmarkJob) cellProfile(c JobCell) (*BaseProfile, error) {
	if c.Profile == "" {
		return j.BaseProfile, nil
	}
	if p := j.findProfile(c.ModelID, c.Profile); p != nil {
		return &BaseProfile{Name: p.Name, Config: p.Config}, nil
	}
	return nil, fmt.Errorf("the job holds no copy of the %q profile for this model — edit the job and choose the profile again", c.Profile)
}

// findProfile returns the job's copy of a model's profile, or nil.
func (j *BenchmarkJob) findProfile(modelID, name string) *JobProfile {
	for i := range j.Profiles {
		if j.Profiles[i].ModelID == modelID && j.Profiles[i].Name == name {
			return &j.Profiles[i]
		}
	}
	return nil
}

// BaseProfile is the saved profile a job measures from: the name, for
// the record, and the whole config, because the fields a snapshot does
// not carry — sampling, jinja, the vision projector — have to come from
// the profile too.
type BaseProfile struct {
	Name   string             `json:"name"`
	Config models.ModelConfig `json:"config"`
}

// IsEmpty reports whether the overrides set nothing: nil, or every field
// nil.
func (o *ConfigOverrides) IsEmpty() bool {
	return overrideKey(o) == ""
}

// ConfigOverrides applies on top of each model's saved ModelConfig for
// every cell. Pointer fields so nil = "use the model's saved value".
type ConfigOverrides struct {
	GPULayers      *int    `json:"gpu_layers,omitempty"`
	ContextSize    *int    `json:"context_size,omitempty"`
	Threads        *int    `json:"threads,omitempty"`
	BatchSize      *int    `json:"batch_size,omitempty"`
	UBatchSize     *int    `json:"ubatch_size,omitempty"`
	CPUMoE         *int    `json:"cpu_moe,omitempty"`
	SplitMode      *string `json:"split_mode,omitempty"`
	FlashAttention *bool   `json:"flash_attention,omitempty"`
	KVCacheQuant   *string `json:"kv_cache_quant,omitempty"`
	DirectIO       *bool   `json:"direct_io,omitempty"`
	PLEMode        *string `json:"ple_mode,omitempty"`
	ExtraFlags     *string `json:"extra_flags,omitempty"`
	GPUAssign      *string `json:"gpu_assign,omitempty"`
	TensorSplit    *string `json:"tensor_split,omitempty"`
	SpecType       *string `json:"spec_type,omitempty"`
	// DraftModelPath is the draft file a cell loads. A sweep value may
	// put a model registry ID here instead of a path; runCell resolves it
	// before the cell runs and before the run records it, because only
	// the runner can look a registry ID up.
	DraftModelPath *string `json:"draft_model_path,omitempty"`
	// DraftKVCacheQuant is the cache type of the drafter's context
	// ("" = llama.cpp's full-precision default).
	DraftKVCacheQuant *string `json:"draft_kv_cache_quant,omitempty"`
	DraftMax          *int    `json:"draft_max,omitempty"`
	DraftMin          *int    `json:"draft_min,omitempty"`
	DraftPMin         *string `json:"draft_p_min,omitempty"`
	SpecAssist        *string `json:"spec_assist,omitempty"`
	AssistNMax        *int    `json:"assist_n_max,omitempty"`
	AssistNMin        *int    `json:"assist_n_min,omitempty"`
	AssistNMatch      *int    `json:"assist_n_match,omitempty"`
	AssistSizeN       *int    `json:"assist_size_n,omitempty"`
	AssistSizeM       *int    `json:"assist_size_m,omitempty"`
	AssistMinHits     *int    `json:"assist_min_hits,omitempty"`
	// Legacy speculative fields. Jobs stored before speculative decoding
	// had two slots still carry these, so the merge keeps honouring them
	// and models.NormalizeSpec moves them onto the assist slot.
	NgramSizeN    *int     `json:"ngram_size_n,omitempty"`
	NgramSizeM    *int     `json:"ngram_size_m,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	MinP          *float64 `json:"min_p,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
}

// JobCell is one (model, build, preset) point in the matrix. The cell
// owns at most one BenchmarkRun at a time; on retry the run ID is
// rewritten to point at the latest attempt.
type JobCell struct {
	ModelID string `json:"model_id"`
	BuildID string `json:"build_id"`
	Preset  string `json:"preset"`
	// Profile is the saved profile this cell measures, whose copy is in
	// the job's Profiles. Empty means the model's current settings.
	Profile        string `json:"profile,omitempty"`
	Status         string `json:"status"` // pending|running|completed|failed|skipped
	Attempt        int    `json:"attempt"`
	BenchmarkRunID string `json:"benchmark_run_id,omitempty"`
	Error          string `json:"error,omitempty"`

	// SweepValues holds this cell's point on each swept axis, keyed by
	// sweep field name. Empty for jobs that sweep nothing. Capability
	// cells carry only the eval-reaching axes (the rest collapse — see
	// ExpandCellsWithSweeps), so this is the complete story of what
	// varied for the cell.
	SweepValues map[string]string `json:"sweep_values,omitempty"`

	// SkipReason explains a cell that completed without producing a run
	// (the KL reference model's own cell — its difference from itself is
	// zero). Rendered wherever cell status renders; empty for every
	// other cell. Deliberately distinct from Error: a skipped-with-reason
	// cell is not a failure, and retry must not re-run it.
	SkipReason string `json:"skip_reason,omitempty"`
}

// newAdhocJob synthesizes the catch-all "Ad-Hoc Runs" pseudo-job that
// holds runs not produced by an explicit batch. Its CreatedAt should
// match the oldest run when a v1 file is being migrated, so the entry
// sorts naturally with the user's history.
func newAdhocJob(createdAt time.Time) BenchmarkJob {
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	return BenchmarkJob{
		ID:        AdhocJobID,
		Name:      "Ad-Hoc Runs",
		Kind:      JobKindAdhoc,
		Status:    JobStatusCompleted,
		CreatedAt: createdAt,
	}
}
