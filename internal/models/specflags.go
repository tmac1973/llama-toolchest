package models

import "strconv"

// specParam is a launch parameter name (without the "--" prefix) and its
// value. Callers format params as CLI flags (EffectiveFlagsFor) or INI
// lines (writeConfigParams).
type specParam struct {
	Name  string
	Value string
}

// appendDraftResourceParams appends the draft-model resource overrides
// shared by the "draft" and "draft-mtp" spec types. Without these, the
// draft model inherits llama-server defaults — which on a large MoE main
// model often means no GPU offload for the draft.
func appendDraftResourceParams(params []specParam, c *ModelConfig) []specParam {
	if c.DraftCtxSize > 0 {
		params = append(params, specParam{"ctx-size-draft", strconv.Itoa(c.DraftCtxSize)})
	}
	if c.DraftGPULayers > 0 {
		params = append(params, specParam{"gpu-layers-draft", strconv.Itoa(c.DraftGPULayers)})
	}
	if c.DraftDevice != "" {
		params = append(params, specParam{"device-draft", c.DraftDevice})
	}
	if c.DraftCPUMoE > 0 {
		params = append(params, specParam{"n-cpu-moe-draft", strconv.Itoa(c.DraftCPUMoE)})
	}
	if c.DraftKVCacheQuant != "" {
		params = append(params,
			specParam{"cache-type-k-draft", c.DraftKVCacheQuant},
			specParam{"cache-type-v-draft", c.DraftKVCacheQuant})
	}
	return params
}

// appendDraftSamplingParams appends the drafting sampling knobs shared by
// the "draft" and "draft-mtp" spec types.
func appendDraftSamplingParams(params []specParam, c *ModelConfig) []specParam {
	if c.DraftMax > 0 {
		params = append(params, specParam{"spec-draft-n-max", strconv.Itoa(c.DraftMax)})
	}
	if c.DraftMin > 0 {
		params = append(params, specParam{"spec-draft-n-min", strconv.Itoa(c.DraftMin)})
	}
	if c.DraftPMin != "" {
		params = append(params, specParam{"spec-draft-p-min", c.DraftPMin})
	}
	return params
}

// specDecodingParams returns the speculative-decoding launch parameters for
// the config. llama.cpp split the legacy mode-agnostic flags (--draft-max,
// --draft-min, --spec-ngram-size-n/m) into mode-specific flags; emit the
// right name based on SpecType.
func specDecodingParams(c *ModelConfig) []specParam {
	var params []specParam
	switch c.SpecType {
	case "draft":
		// Internal value is still "draft" (legacy config compat) but
		// llama.cpp renamed the spec-type enum value to "draft-simple"
		// alongside the introduction of "draft-mtp" and "draft-eagle3".
		// Without --spec-type, the draft model loads but isn't used —
		// the default is "none".
		params = append(params, specParam{"spec-type", "draft-simple"})
		if c.DraftModelPath != "" {
			params = append(params, specParam{"model-draft", c.DraftModelPath})
		}
		params = appendDraftSamplingParams(params, c)
		params = appendDraftResourceParams(params, c)
	case "draft-mtp":
		// Two MTP flavors share this spec-type:
		//   • Self-speculation (Qwen3.6, DeepSeek-V3): the MTP head is baked
		//     into the main GGUF, so MtpPath is empty — no --model-draft and
		//     the draft-resource flags don't apply. Only the sampling knobs do.
		//   • Separate drafter (gemma-4's "gemma4-assistant" head): the head
		//     ships as its own GGUF in MtpPath, loaded via --model-draft just
		//     like a draft-simple model, including the draft-resource overrides.
		params = append(params, specParam{"spec-type", "draft-mtp"})
		if c.MtpPath != "" && !c.MtpDisabled {
			params = append(params, specParam{"model-draft", c.MtpPath})
			params = appendDraftResourceParams(params, c)
		}
		params = appendDraftSamplingParams(params, c)
	case "ngram-mod":
		params = append(params, specParam{"spec-type", c.SpecType})
		if c.DraftMax > 0 {
			params = append(params, specParam{"spec-ngram-mod-n-max", strconv.Itoa(c.DraftMax)})
		}
		if c.DraftMin > 0 {
			params = append(params, specParam{"spec-ngram-mod-n-min", strconv.Itoa(c.DraftMin)})
		}
	case "ngram-simple", "ngram-map-k", "ngram-map-k4v":
		params = append(params, specParam{"spec-type", c.SpecType})
		prefix := "spec-" + c.SpecType
		if c.NgramSizeN > 0 {
			params = append(params, specParam{prefix + "-size-n", strconv.Itoa(c.NgramSizeN)})
		}
		if c.NgramSizeM > 0 {
			params = append(params, specParam{prefix + "-size-m", strconv.Itoa(c.NgramSizeM)})
		}
	case "ngram-cache":
		params = append(params, specParam{"spec-type", c.SpecType})
	}
	return params
}
