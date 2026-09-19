package api

// fieldHelp explains each setting the autoconfigure review shows, in the
// same words as the config form's tooltips, keyed by ModelConfig JSON
// name.
var fieldHelp = map[string]string{
	"context_size":     "The longest conversation or document the model can hold at once, in tokens (about three quarters of a word each). Larger values use more GPU memory.",
	"kv_cache_quant":   "How the conversation memory (KV cache) is stored. f16 is full precision. q8_0 uses about half the memory with a very small effect on quality; q4_0 uses about a quarter and may affect quality.",
	"gpu_layers":       "How many of the model's layers run on the GPU. All is fastest; layers left on the CPU are much slower.",
	"cpu_moe":          "Mixture-of-experts models only: how many layers keep their expert weights in system memory instead of GPU memory. Lets a model larger than the GPU run, at some cost in speed.",
	"gpu_assign":       "Which GPUs the model is spread over.",
	"flash_attention":  "A faster way to compute attention that also uses less memory. Recommended for most models.",
	"ubatch_size":      "How many prompt tokens are processed at once. Larger values read long prompts faster but use more GPU memory. Autotune can measure the best values for this machine.",
	"parallel":         "How many conversations the model serves at the same time. Each one gets an equal share of the context.",
	"threads":          "CPU threads used for the parts of the model that run on the CPU. One per physical core is usually best.",
	"spec_type":        "Speculative decoding: a fast way to guess several tokens ahead and check them in one step. It speeds up generation and does not change the answers.",
	"temperature":      "How random the answers are. Lower is more focused and repeatable; higher is more varied.",
	"top_p":            "Only tokens within this share of the total probability are considered. Lower is more focused.",
	"top_k":            "Only the this-many most likely tokens are considered at each step.",
	"min_p":            "Tokens much less likely than the most likely one are left out. 0 turns this off.",
	"presence_penalty": "Pushes the model toward new topics by penalizing tokens it has already used.",
	"repeat_penalty":   "Discourages repeating the same words. 1.0 turns this off.",
}
