package api

// fieldHelp explains each setting the autoconfigure review shows, in the
// same words as the config form's tooltips, keyed by ModelConfig JSON
// name.
var fieldHelp = map[string]string{
	"context_size":      "The longest conversation or document the model can hold at once, in tokens (about three quarters of a word each). Larger values use more GPU memory.",
	"kv_cache_quant":    "How the conversation memory (KV cache) is stored. f16 is full precision. q8_0 uses about half the memory with a very small effect on quality; q4_0 uses about a quarter and may affect quality. f8_e4m3 (llama.cpp b12274 and later) is an 8-bit floating-point cache, slightly smaller than q8_0.",
	"gpu_layers":        "How many of the model's layers run on the GPU. All is fastest; layers left on the CPU are much slower.",
	"cpu_moe":           "Mixture-of-experts models only: how many layers keep their expert weights in system memory instead of GPU memory. Lets a model larger than the GPU run, at some cost in speed.",
	"moe_cache_mib":     "Mixture-of-experts models only: a GPU cache, in MiB, for the expert weights kept in system memory. The experts tokens use most often are then read from the GPU instead. Uses GPU memory left over after the model.",
	"loop_core_one_gpu": "The layers this model runs several times per token. On one GPU it is much faster. That GPU holds the most, so less context fits.",
	"mmproj_in_ram":     "Where the vision projector, which reads images and audio, is kept. On the GPU it is faster; in system memory its GPU memory goes to the model, and images are read on the CPU.",
	"gpu_assign":        "Which GPUs the model is spread over.",
	"flash_attention":   "A faster way to compute attention that also uses less memory. Recommended for most models.",
	"ubatch_size":       "How many prompt tokens are processed at once. Larger values read long prompts faster but use more GPU memory. Autotune can measure the best values for this machine.",
	"parallel":          "How many conversations the model serves at the same time. By default llama.cpp serves 4 that share the whole context; from 2, each one gets an equal share unless Shared Context is on.",
	"threads":           "CPU threads used for the parts of the model that run on the CPU. One per physical core is usually best.",
	"spec_type":         "Speculative decoding: a fast way to guess several tokens ahead and check them in one step. It speeds up generation and does not change the answers.",
	"temperature":       "How random the answers are. Lower is more focused and repeatable; higher is more varied.",
	"top_p":             "Only tokens within this share of the total probability are considered. Lower is more focused.",
	"top_k":             "Only the this-many most likely tokens are considered at each step.",
	"min_p":             "Tokens much less likely than the most likely one are left out. 0 turns this off.",
	"presence_penalty":  "Pushes the model toward new topics by penalizing tokens it has already used.",
	"repeat_penalty":    "Discourages repeating the same words. 1.0 turns this off.",
}

// sharedFieldHelp replaces fieldHelp entries on unified memory (Apple
// Silicon), where nothing moves to other memory: the CPU runs what the
// GPU's share of the memory has no room for.
var sharedFieldHelp = map[string]string{
	"cpu_moe":       "Mixture-of-experts models only: how many layers have their expert weights run by the CPU instead of the GPU. The weights stay in the same memory. Lets a model larger than the GPU's share of the memory run, at some cost in speed.",
	"moe_cache_mib": "Mixture-of-experts models only: GPU memory, in MiB, for copies of the expert weights the CPU runs. The GPU then runs the experts tokens use most often. On a Mac each copy uses the memory a second time.",
	"mmproj_in_ram": "Where the vision projector, which reads images and audio, runs. On the GPU it is faster; on the CPU its share of the GPU's memory goes to the model, and images are read more slowly.",
	"threads":       "CPU threads used for the parts of the model that run on the CPU. One per performance core is usually best.",
}

// fieldHelpFor is the help for a setting, in the words for unified
// memory when shared is set.
func fieldHelpFor(key string, shared bool) string {
	if h, ok := sharedFieldHelp[key]; ok && shared {
		return h
	}
	return fieldHelp[key]
}
