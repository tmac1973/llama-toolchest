package models

import (
	"strconv"
	"strings"
)

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

// draftTypeName maps a config's draft mode onto llama.cpp's own
// --spec-type value. Only "draft" differs: llama.cpp renamed that enum
// value to "draft-simple" alongside the introduction of "draft-mtp" and
// "draft-eagle3", after configs had already been saved with the old one.
func draftTypeName(mode string) string {
	if mode == "draft" {
		return "draft-simple"
	}
	return mode
}

// appendDraftParams appends the flags of the draft-method slot: the
// drafter to load, its resource overrides, and its sampling knobs.
func appendDraftParams(params []specParam, c *ModelConfig) []specParam {
	// Where the drafter comes from depends on the mode:
	//   • Self-speculation MTP (Qwen3.5/3.6, DeepSeek-V3): the head is
	//     baked into the main GGUF, so MtpPath is empty — no
	//     --model-draft, and the draft-resource flags don't apply.
	//   • Separate MTP drafter (gemma-4's "gemma4-assistant" head): the
	//     head ships as its own GGUF in MtpPath.
	//   • Every other draft method: DraftModelPath, whether that is a
	//     smaller model of the same family (draft) or a converted head
	//     (draft-eagle3, draft-dflash, draft-dspark).
	drafter := c.DraftModelPath
	if c.SpecType == "draft-mtp" {
		drafter = ""
		if c.MtpPath != "" && !c.MtpDisabled {
			drafter = c.MtpPath
		}
	}
	if drafter != "" {
		params = append(params, specParam{"model-draft", drafter})
		params = appendDraftResourceParams(params, c)
	}
	return appendDraftSamplingParams(params, c)
}

// appendAssistParams appends the flags of the draftless n-gram slot.
// The modes do not share a parameter set: ngram-mod is tuned by draft
// lengths and a match length, the map/simple family by lookup and draft
// sizes and a hit threshold, and ngram-cache takes nothing at all.
func appendAssistParams(params []specParam, c *ModelConfig) []specParam {
	switch c.SpecAssist {
	case "ngram-mod":
		if c.AssistNMax > 0 {
			params = append(params, specParam{"spec-ngram-mod-n-max", strconv.Itoa(c.AssistNMax)})
		}
		if c.AssistNMin > 0 {
			params = append(params, specParam{"spec-ngram-mod-n-min", strconv.Itoa(c.AssistNMin)})
		}
		if c.AssistNMatch > 0 {
			params = append(params, specParam{"spec-ngram-mod-n-match", strconv.Itoa(c.AssistNMatch)})
		}
	case "ngram-simple", "ngram-map-k", "ngram-map-k4v":
		prefix := "spec-" + c.SpecAssist
		if c.AssistSizeN > 0 {
			params = append(params, specParam{prefix + "-size-n", strconv.Itoa(c.AssistSizeN)})
		}
		if c.AssistSizeM > 0 {
			params = append(params, specParam{prefix + "-size-m", strconv.Itoa(c.AssistSizeM)})
		}
		if c.AssistMinHits > 0 {
			params = append(params, specParam{prefix + "-min-hits", strconv.Itoa(c.AssistMinHits)})
		}
	}
	return params
}

// specDecodingParams returns the speculative-decoding launch parameters for
// the config: one --spec-type carrying the comma-separated list of active
// modes, then each slot's own flags. llama.cpp split the legacy
// mode-agnostic flags (--draft-max, --draft-min, --spec-ngram-size-n/m)
// into per-mode ones, so the two slots' draft lengths coexist —
// common_speculative_n_max takes the maximum over whatever is enabled.
//
// The list is written as one value rather than a repeated flag on
// purpose: common/preset.cpp parses an INI section into a map, so a
// second "spec-type =" line would silently replace the first.
func specDecodingParams(c *ModelConfig) []specParam {
	// Copied, not mutated in place: this runs on the registry's own
	// struct, and normalising a config that still holds the pre-split
	// shape must not write back to it.
	cfg := *c
	NormalizeSpec(&cfg)

	var modes []string
	var params []specParam
	if IsDraftMode(cfg.SpecType) {
		modes = append(modes, draftTypeName(cfg.SpecType))
		params = appendDraftParams(params, &cfg)
	}
	if IsAssistMode(cfg.SpecAssist) {
		modes = append(modes, cfg.SpecAssist)
		params = appendAssistParams(params, &cfg)
	}
	if len(modes) == 0 {
		// Nothing, not "none": common_speculative_types_from_names
		// discards every other entry in a list the moment it sees
		// "none", so an empty slot must contribute no name at all.
		return nil
	}
	return append([]specParam{{"spec-type", strings.Join(modes, ",")}}, params...)
}
