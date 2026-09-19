package benchmark

import "github.com/tmac1973/llama-toolchest/internal/models"

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
