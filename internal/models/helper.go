package models

import "runtime"

// Helper models are the app's own: small models it loads to read a model
// card for Autoconfigure, and unloads afterwards. Their settings are not
// the user's to choose — a helper with a context too small to hold a model
// card, or with half its layers on the CPU, only produces confusing
// failures in a feature that is supposed to remove confusion.
//
// So the settings are fixed here, and recomputed before every use. The
// only choice left is the context, which is sized to the GPU: enough for
// a long model card where there is room, less where there is not.
const (
	// HelperContextFull holds the longest model cards comfortably.
	HelperContextFull = 32768
	// HelperContextNormal is the target: a trimmed card plus the answer.
	HelperContextNormal = 16384
	// HelperContextSmall is the floor, for a small or busy GPU. Cards are
	// trimmed harder to match (see autoconfig.CardCharsForContext).
	HelperContextSmall = 8192
)

// HelperConfig is the launch config for a helper model on a machine with
// budgetGiB of GPU memory to spare. Everything is on the GPU, at
// llama.cpp's own batch sizes, with no speculative decoding and no
// sampling settings: the requests set what they need.
//
// A budget of 0 (unknown hardware) uses the normal context.
func HelperConfig(m *Model, budgetGiB float64) ModelConfig {
	cfg := ModelConfig{
		Enabled:        true,
		GPULayers:      999,
		ContextSize:    HelperContextNormal,
		Threads:        ThreadsFor(runtime.NumCPU()),
		FlashAttention: true,
		Jinja:          true,
	}
	if m != nil && m.ContextLength > 0 && m.ContextLength < cfg.ContextSize {
		cfg.ContextSize = m.ContextLength
	}
	if budgetGiB <= 0 || m == nil {
		return cfg
	}
	// The largest of the three that fits, never below the floor: a
	// helper that does not fit on the GPU would be slower than the
	// feature is worth, and one with no room for a card is useless.
	for _, ctx := range []int{HelperContextFull, HelperContextNormal, HelperContextSmall} {
		if m.ContextLength > 0 && ctx > m.ContextLength {
			continue
		}
		try := cfg
		try.ContextSize = ctx
		if VRAMEstimateForConfigOn(m, &try, 1) <= budgetGiB {
			return try
		}
	}
	cfg.ContextSize = HelperContextSmall
	if m.ContextLength > 0 && m.ContextLength < cfg.ContextSize {
		cfg.ContextSize = m.ContextLength
	}
	return cfg
}
