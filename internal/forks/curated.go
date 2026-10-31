package forks

// Curated returns the curated forks, in the order the UI lists them.
func Curated() []Fork {
	return []Fork{avalon()}
}

// Find returns the curated fork with the given ID.
func Find(id string) (Fork, bool) {
	for _, f := range Curated() {
		if f.ID == id {
			return f, true
		}
	}
	return Fork{}, false
}

const avalonModelFile = "Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf"
const avalonDraftFile = "MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf"

// avalon is freriadoc/Avalon, a fork tuned for Qwen3.8-Flash-Next (a
// 125B MoE) on one RDNA3 GPU: the dense weights, the KV cache and a cache
// of the hottest experts stay in VRAM, the other experts in system RAM.
// Its GGUF uses quant types upstream llama.cpp does not have, so the
// model only runs on this fork.
func avalon() Fork {
	return Fork{
		ID:         "avalon",
		Name:       "Avalon",
		Repo:       "https://github.com/freriadoc/Avalon",
		PinnedSHA:  "ad2c6e750c7e1fb59363bff7d5a58e4794603c13",
		PinnedDate: "2026-10-31",
		Backends:   []string{"rocm", "cuda"},
		// The launch command of Avalon's README. --fit, on by default,
		// puts the dense weights on the GPU and the experts in RAM, and
		// the expert cache takes the rest of the VRAM; --parallel 1 keeps
		// other slots from holding part of the KV cache.
		Recipe: Recipe{
			Args: []string{
				"-c", "262144", "-b", "9536", "-ub", "9536",
				"-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0",
				"--jinja", "--parallel", "1",
				"--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
				"-ubd", "256", "-ctkd", "q8_0", "-ctvd", "q8_0",
			},
			ModelFileArg: avalonModelFile,
			DraftFileArg: avalonDraftFile,
		},
		ModelRepo: "freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF",
		ModelName: "Qwen3.8-Flash-Next Avalon IQ3_KS",
		// Sizes from the Hugging Face tree API on 2026-10-31.
		ModelFiles: []ModelFile{
			{avalonModelFile, 693388160},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00002-of-00033.gguf", 38400184512},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00003-of-00033.gguf", 1959834208},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00004-of-00033.gguf", 1905877888},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00005-of-00033.gguf", 1111012160},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00006-of-00033.gguf", 2170465728},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00007-of-00033.gguf", 1188551424},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00008-of-00033.gguf", 2008170208},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00009-of-00033.gguf", 1948396000},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00010-of-00033.gguf", 1174691616},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00011-of-00033.gguf", 1786100480},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00012-of-00033.gguf", 2336862944},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00013-of-00033.gguf", 1310512704},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00014-of-00033.gguf", 2441460480},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00015-of-00033.gguf", 2087073504},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00016-of-00033.gguf", 1127764544},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00017-of-00033.gguf", 2266317792},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00018-of-00033.gguf", 2353966592},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00019-of-00033.gguf", 1188551488},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00020-of-00033.gguf", 2466922240},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00021-of-00033.gguf", 2276076032},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00022-of-00033.gguf", 1318870816},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00023-of-00033.gguf", 2257959680},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00024-of-00033.gguf", 2376184544},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00025-of-00033.gguf", 930403904},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00026-of-00033.gguf", 2402138880},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00027-of-00033.gguf", 2362324704},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00028-of-00033.gguf", 1246833280},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00029-of-00033.gguf", 1325736544},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00030-of-00033.gguf", 2420606496},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00031-of-00033.gguf", 1286154880},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00032-of-00033.gguf", 1325736544},
			{"Qwen3.8-Flash-Next-Avalon-IQ3KS-00033-of-00033.gguf", 1391129888},
			{avalonDraftFile, 1484237088},
			// The expert ranking the cache is filled from at load; Avalon
			// reads it from beside the model.
			{"moe-cache-ranking.bin", 177192},
		},
		License: "MIT (Avalon); Qwen Community License 1.0 (model)",
		Notes:   "Runs Qwen3.8-Flash-Next on one RDNA3 GPU with the experts in system RAM. Tuned on an RX 7900 XT with ROCm 7.2.",
		Needs: Needs{
			MinRAMGiB:  32,
			AVX2:       true,
			ROCmArchs:  []string{"gfx1100", "gfx1101", "gfx1102"},
			CUDA:       true,
			MinDiskGiB: 98,
		},
		// The thinking-mode values on the model card,
		// https://huggingface.co/Qwen/Qwen3.8-Flash-Next.
		SamplingDefaults: map[string]any{
			"temperature": 1.0,
			"top_p":       0.95,
			"top_k":       20,
			"min_p":       0.0,
		},
	}
}
