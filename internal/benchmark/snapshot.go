package benchmark

import (
	"fmt"
	"path/filepath"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// SnapshotFromConfig records the benchmark-relevant part of a model's
// launch config: every field that changes what llama-server loads, and so
// the numbers a run measures. Sampling, the chat template, aliases and the
// like are left out: they do not change throughput, and every field here
// is a dimension the comparison view can split runs by.
//
// profile and edited are the model's active saved profile and whether the
// config was changed since it (models.Registry.ActiveProfileState).
func SnapshotFromConfig(cfg models.ModelConfig, profile string, edited bool) ConfigSnapshot {
	return ConfigSnapshot{
		GPULayers:         cfg.GPULayers,
		ContextSize:       cfg.ContextSize,
		GPUAssign:         cfg.GPUAssign,
		TensorSplit:       cfg.TensorSplit,
		FlashAttention:    cfg.FlashAttention,
		KVCacheQuant:      cfg.KVCacheQuant,
		DirectIO:          cfg.DirectIO,
		Threads:           cfg.Threads,
		BatchSize:         cfg.BatchSize,
		UBatchSize:        cfg.UBatchSize,
		SpecType:          cfg.SpecType,
		DraftModelPath:    cfg.DraftModelPath,
		MtpPath:           cfg.MtpPath,
		DraftMax:          cfg.DraftMax,
		DraftMin:          cfg.DraftMin,
		DraftPMin:         cfg.DraftPMin,
		SpecAssist:        cfg.SpecAssist,
		AssistNMax:        cfg.AssistNMax,
		AssistNMin:        cfg.AssistNMin,
		AssistNMatch:      cfg.AssistNMatch,
		AssistSizeN:       cfg.AssistSizeN,
		AssistSizeM:       cfg.AssistSizeM,
		AssistMinHits:     cfg.AssistMinHits,
		NgramSizeN:        cfg.NgramSizeN,
		NgramSizeM:        cfg.NgramSizeM,
		PLEMode:           cfg.PLEMode,
		ExtraFlags:        cfg.ExtraFlags,
		SplitMode:         cfg.SplitMode,
		MainGPU:           cfg.MainGPU,
		Parallel:          cfg.Parallel,
		CPUMoE:            cfg.CPUMoE,
		DraftCtxSize:      cfg.DraftCtxSize,
		DraftGPULayers:    cfg.DraftGPULayers,
		DraftKVCacheQuant: cfg.DraftKVCacheQuant,
		ProfileName:       profile,
		ProfileEdited:     edited,
	}
}

// markProfileEdited sets ProfileEdited on a cell's config when it differs
// from the model's config it started from, in any field: a job override,
// a swept value, or an evaluation's default KV cache. A run that is
// already marked (the live config was changed since its profile) stays
// marked.
func markProfileEdited(cfg, base ConfigSnapshot) ConfigSnapshot {
	cmp := cfg
	cmp.ProfileName, cmp.ProfileEdited = base.ProfileName, base.ProfileEdited
	cfg.ProfileEdited = base.ProfileEdited || cmp != base
	return cfg
}

// ApplySnapshotToConfig overlays a benchmark ConfigSnapshot onto a copy
// of the model's saved config. Zero values mean "not overridden", which
// matches how ConfigSnapshot is built in ResolveModel — a snapshot
// always carries the saved value unless a job override replaced it.
func ApplySnapshotToConfig(base models.ModelConfig, snap ConfigSnapshot) models.ModelConfig {
	out := base

	// Every field is assigned unconditionally. ResolveModel seeds the
	// snapshot from the model's saved config and applyOverrides then
	// replaces only what the job set, so the snapshot is authoritative
	// for every field it models — a zero is the value zero, not "unset".
	//
	// Skipping zeros here silently discarded legitimate overrides
	// (gpu_layers=0 for CPU-only, ubatch/threads/context edge values)
	// while the run still recorded them as applied. That is exactly the
	// mislabeled-result bug this whole mechanism exists to prevent.
	out.GPULayers = snap.GPULayers
	out.ContextSize = snap.ContextSize
	out.Threads = snap.Threads
	out.BatchSize = snap.BatchSize
	out.UBatchSize = snap.UBatchSize
	out.CPUMoE = snap.CPUMoE
	out.GPUAssign = snap.GPUAssign
	out.TensorSplit = snap.TensorSplit
	out.KVCacheQuant = snap.KVCacheQuant
	out.SpecType = snap.SpecType
	out.DraftModelPath = snap.DraftModelPath
	out.DraftMax = snap.DraftMax
	out.DraftMin = snap.DraftMin
	out.DraftPMin = snap.DraftPMin
	out.SpecAssist = snap.SpecAssist
	out.AssistNMax = snap.AssistNMax
	out.AssistNMin = snap.AssistNMin
	out.AssistNMatch = snap.AssistNMatch
	out.AssistSizeN = snap.AssistSizeN
	out.AssistSizeM = snap.AssistSizeM
	out.AssistMinHits = snap.AssistMinHits
	out.NgramSizeN = snap.NgramSizeN
	out.NgramSizeM = snap.NgramSizeM
	out.FlashAttention = snap.FlashAttention
	out.DirectIO = snap.DirectIO
	out.PLEMode = snap.PLEMode
	out.ExtraFlags = snap.ExtraFlags
	// Placement as launched, the MoE offload, the parallel slots and the
	// draft model's own resources.
	//
	// SplitMode and MainGPU are copied only when the snapshot has them:
	// they are derived from the GPU assignment when a config is saved,
	// and runs recorded before the snapshot carried them have neither.
	// Copying a blank would erase the placement the config derived.
	if snap.SplitMode != "" {
		out.SplitMode = snap.SplitMode
	}
	if snap.MainGPU != 0 {
		out.MainGPU = snap.MainGPU
	}
	out.Parallel = snap.Parallel
	out.CPUMoE = snap.CPUMoE
	out.MtpPath = snap.MtpPath
	out.DraftCtxSize = snap.DraftCtxSize
	out.DraftGPULayers = snap.DraftGPULayers
	out.DraftKVCacheQuant = snap.DraftKVCacheQuant
	return out
}

// ConfigForValues is the launch config a set of sweep values produces on
// top of base: what a cell actually ran, and so what a profile saving
// that cell's settings must hold. resolve turns a draft file named by a
// registry ID into its path; it may be nil when no value names one.
func ConfigForValues(base models.ModelConfig, values map[string]string, resolve func(id string) (string, error)) (models.ModelConfig, error) {
	ov, err := CellOverrides(nil, values)
	if err != nil {
		return base, err
	}
	snap := SnapshotFromConfig(base, "", false)
	out := ApplySnapshotToConfig(base, ApplyOverrides(snap, ov))
	// Only a draft file these values named is resolved; the base config's
	// own draft model stays where it is.
	if ov != nil && ov.DraftModelPath != nil {
		if err := ResolveDraftFile(&out, resolve); err != nil {
			return base, err
		}
	}
	return out, nil
}

// ResolveDraftFile turns the draft file a spec value named — a model
// registry ID, or a path — into the field the launch actually reads:
// MtpPath for draft-mtp, which loads a head, and DraftModelPath for every
// other draft method. A value that is already a path is used as it is.
func ResolveDraftFile(cfg *models.ModelConfig, resolve func(id string) (string, error)) error {
	raw := cfg.DraftModelPath
	if raw == "" {
		return nil
	}
	path := raw
	if !filepath.IsAbs(raw) {
		if resolve == nil {
			return fmt.Errorf("draft model %q cannot be resolved here", raw)
		}
		p, err := resolve(raw)
		if err != nil || p == "" {
			return fmt.Errorf("draft model %q is not installed", raw)
		}
		path = p
	}
	if cfg.SpecType == "draft-mtp" {
		// draft-mtp loads a head through MtpPath; DraftModelPath is for
		// the methods that load a whole model.
		cfg.MtpPath, cfg.DraftModelPath = path, ""
		return nil
	}
	cfg.DraftModelPath = path
	return nil
}
