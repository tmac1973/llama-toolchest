package models

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// GGUFMeta holds architecture parameters extracted from a GGUF file header.
type GGUFMeta struct {
	Architecture  string `json:"architecture"`
	NLayers       int    `json:"n_layers"`
	NEmbd         int    `json:"n_embd"`
	NHead         int    `json:"n_head"`
	NKVHead       int    `json:"n_kv_head"`
	ContextLength int    `json:"context_length"` // max trained context size
	// VocabSize is the tokenizer's vocabulary count, read from the LENGTH
	// of the tokenizer.ggml.tokens array in the header (the array lengths
	// sit in the header; the contents are seek-past, not read). Zero on
	// files without the key and on records parsed before this field
	// existed; consumers that need a value anyway apply their own
	// documented fallback (the KL logits estimate in internal/evaluate).
	VocabSize     int  `json:"vocab_size,omitempty"`
	SupportsTools bool `json:"supports_tools"` // chat template references tools
	HasVision     bool `json:"has_vision"`     // model has a built-in vision encoder

	// Reasoning / thinking mode detected from the chat template (see
	// detectReasoning). Reasoning is the assembled capability; ReasoningChecked
	// records that detection ran, so a false Supported on an old registry record
	// can be told apart from "never inspected" during backfill.
	Reasoning        ReasoningCapability `json:"reasoning"`
	ReasoningChecked bool                `json:"reasoning_checked"`

	// KV-cache scaling factors, precomputed per-layer to capture grouped-query
	// attention (which can vary per layer) and sliding-window attention. The KV
	// cache at context C is: (C·KVFullPerTok + min(C,SlidingWindow)·KVSWAPerTok)
	// × bytes_per_element. "PerTok" values are KV elements per token, already
	// summed over the relevant layers and including both K and V.
	KVFullPerTok  int `json:"kv_full_per_tok,omitempty"` // Σ full-attention layers of kv_heads·(k_dim+v_dim)
	KVSWAPerTok   int `json:"kv_swa_per_tok,omitempty"`  // Σ sliding-window layers of kv_heads·(k_dim_swa+v_dim_swa)
	SlidingWindow int `json:"sliding_window,omitempty"`  // sliding-window size; 0 if no local-attention layers

	// Author-recommended sampling defaults from general.sampling.* keys
	// (llama.cpp PR #17120; convert_hf_to_gguf.py fills them from the upstream
	// generation_config.json). Nil = key absent. SamplingChecked mirrors
	// ReasoningChecked: it lets backfill tell "no keys in the file" apart from
	// "parsed before these keys were read".
	SamplingTemp          *float64 `json:"sampling_temp,omitempty"`
	SamplingTopP          *float64 `json:"sampling_top_p,omitempty"`
	SamplingTopK          *int     `json:"sampling_top_k,omitempty"`
	SamplingMinP          *float64 `json:"sampling_min_p,omitempty"`
	SamplingRepeatPenalty *float64 `json:"sampling_repeat_penalty,omitempty"`
	SamplingChecked       bool     `json:"sampling_checked"`

	// PLEBytes is the on-disk size of per_layer_token_embd.weight, the
	// per-layer / n-gram embedding table carried by Gemma-3N, Gemma-4 and
	// Qwen4-Exp. Zero when the file has no such tensor, which is the case
	// for every other architecture. It is read from the tensor-info block
	// rather than inferred from the architecture name, so a new model that
	// adopts the same table is picked up without a code change.
	//
	// The table is the reason this parser looks past the metadata at all:
	// it can be tens of gigabytes (97.7 GiB unquantized on
	// Qwen3.8-Flash-Next), and llama.cpp reads its rows on demand from the
	// mmap rather than making it resident. Counting it as VRAM, as a plain
	// file-size estimate does, overstates the requirement by more than the
	// rest of the model put together.
	PLEBytes int64 `json:"ple_bytes,omitempty"`
	// PLEChecked mirrors ReasoningChecked and the rest: it records that
	// the tensor table was inspected, so a record written before split
	// models were scanned correctly can be told apart from one whose file
	// genuinely has no table.
	PLEChecked bool `json:"ple_checked,omitempty"`
	// TokenEmbdBytes is the on-disk size of token_embd.weight, which
	// llama.cpp holds host-mapped rather than on a device. Measured from
	// the tensor block for the same reason as PLEBytes: publishers quantize
	// it independently of the body.
	TokenEmbdBytes int64 `json:"token_embd_bytes,omitempty"`
	// IndexerKeyLength is attention.indexer.key_length, present only on
	// sparse-attention architectures. Non-zero means the model ranks the
	// whole cache before attending, which costs both a key cache of its
	// own and graph scratch proportional to context x micro-batch.
	IndexerKeyLength int `json:"indexer_key_length,omitempty"`
	// AttnLayers counts the layers that actually hold a KV cache. Equal to
	// NLayers on a plain model; smaller on a hybrid, where the rest carry
	// recurrent state.
	AttnLayers int `json:"attn_layers,omitempty"`
	// KVRecurrentChecked records that the KV factors were computed by a
	// parser that knows recurrent layers hold no cache. Records written
	// before that have plausible non-zero factors which are simply too
	// large, so "is it zero" cannot tell them apart — only this can.
	KVRecurrentChecked bool `json:"kv_recurrent_checked,omitempty"`

	// BaseModelRepo is the upstream "org/repo" this quant derives from, per
	// general.base_model.0.repo_url. Used to locate the base model's
	// generation_config.json when the file has no embedded sampling keys.
	BaseModelRepo string `json:"base_model_repo,omitempty"`
}

// ApplyTo copies parsed GGUF metadata onto a Model.
func (meta *GGUFMeta) ApplyTo(m *Model) {
	m.Arch = meta.Architecture
	m.NLayers = meta.NLayers
	m.NEmbd = meta.NEmbd
	m.NHead = meta.NHead
	m.NKVHead = meta.NKVHead
	m.ContextLength = meta.ContextLength
	// A parse always looks for tokenizer.ggml.tokens, so the question is
	// answered either way — record that, and only overwrite the value
	// when there was one, so a parse that found nothing cannot clear a
	// size an earlier one established.
	m.VocabChecked = true
	if meta.VocabSize > 0 {
		m.VocabSize = meta.VocabSize
	}
	m.SupportsTools = meta.SupportsTools
	m.HasBuiltinVision = meta.HasVision
	m.Reasoning = meta.Reasoning
	m.ReasoningChecked = meta.ReasoningChecked
	m.KVFullPerTok = meta.KVFullPerTok
	m.KVSWAPerTok = meta.KVSWAPerTok
	m.SlidingWindow = meta.SlidingWindow
	m.SamplingChecked = meta.SamplingChecked
	m.PLEBytes = meta.PLEBytes
	m.PLEChecked = meta.PLEChecked
	m.KVRecurrentChecked = meta.KVRecurrentChecked
	m.TokenEmbdBytes = meta.TokenEmbdBytes
	m.IndexerKeyLength = meta.IndexerKeyLength
	m.AttnLayers = meta.AttnLayers
	if meta.BaseModelRepo != "" {
		m.BaseModelRepo = meta.BaseModelRepo
	}
	// Merge rather than assign: re-parses must not drop presets gathered from
	// network sources after download.
	if p := meta.EmbeddedSamplingPreset(); p != nil {
		m.SamplingPresets = UpsertSamplingPreset(m.SamplingPresets, *p)
	}
}

// EmbeddedSamplingPreset assembles the general.sampling.* values into a
// preset, or nil when the file carries none. llama-server applies these same
// values on its own when launched without explicit sampling flags, so this
// preset surfaces what is already the effective default rather than changing
// behavior.
func (meta *GGUFMeta) EmbeddedSamplingPreset() *SamplingPreset {
	if meta.SamplingTemp == nil && meta.SamplingTopP == nil && meta.SamplingTopK == nil &&
		meta.SamplingMinP == nil && meta.SamplingRepeatPenalty == nil {
		return nil
	}
	return &SamplingPreset{
		Name:          "default",
		Label:         "Model-embedded default",
		Description:   "From GGUF metadata (general.sampling.*) — active whenever no override is set",
		Source:        "gguf",
		Temperature:   meta.SamplingTemp,
		TopP:          meta.SamplingTopP,
		TopK:          meta.SamplingTopK,
		MinP:          meta.SamplingMinP,
		RepeatPenalty: meta.SamplingRepeatPenalty,
	}
}

// HeadDim returns the dimension per attention head.
func (m *GGUFMeta) HeadDim() int {
	if m.NHead == 0 {
		return 0
	}
	return m.NEmbd / m.NHead
}

// ParseGGUFMeta reads architecture metadata from a GGUF file.
// Only reads the header — does not load tensors or weights.
func ParseGGUFMeta(path string) (*GGUFMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	meta, err := ParseGGUFMetaFrom(f)
	if err != nil {
		return nil, err
	}
	// A split model keeps its tensor table in a later shard; see
	// scanShardsForPLE.
	if meta.PLEBytes == 0 || meta.TokenEmbdBytes == 0 {
		ple, emb := scanShardsForTensors(path)
		if meta.PLEBytes == 0 {
			meta.PLEBytes = ple
		}
		if meta.TokenEmbdBytes == 0 {
			meta.TokenEmbdBytes = emb
		}
	}
	return meta, nil
}

// ParseGGUFMetaFrom reads the same metadata from an arbitrary seekable
// source. Split out from ParseGGUFMeta so a GGUF that isn't on disk yet
// can be inspected over ranged HTTP reads — see modelsource.ProbePLE,
// which uses it to answer "how much of this download never reaches VRAM"
// before anyone commits to the download.
func ParseGGUFMetaFrom(f io.ReadSeeker) (*GGUFMeta, error) {

	// Magic: "GGUF"
	var magic [4]byte
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file (magic: %q)", magic)
	}

	// Version
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	// Tensor count and metadata KV count
	var tensorCount, kvCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("read kv count: %w", err)
	}

	meta := &GGUFMeta{}

	// GGUF's documented default when general.alignment is absent.
	alignment := int64(32)

	// Locals gathered across the metadata scan, reduced into the KV-cache
	// scaling factors once architecture-prefixed keys are all read. The full
	// scan (no early exit) is cheap: values we don't capture are seek-skipped.
	var (
		headCountKV   int   // scalar head_count_kv (0 if absent or array form)
		kvHeadCounts  []int // per-layer head_count_kv (gemma stores this as an array)
		keyLen        int   // attention.key_length (explicit head dim for K)
		valLen        int   // attention.value_length
		keyLenSWA     int   // attention.key_length_swa (sliding-window layers)
		valLenSWA     int   // attention.value_length_swa
		slidingWindow int   // attention.sliding_window size
		swaPattern    []bool
		// Hybrid models interleave recurrent (linear-attention) layers,
		// which hold no KV cache at all. Either key identifies them; the
		// explicit array wins when both are present, matching llama.cpp.
		fullAttnInterval int    // full_attention_interval
		recurrentLayers  []bool // attention.recurrent_layers
	)

	for i := uint64(0); i < kvCount; i++ {
		key, err := readGGUFString(f)
		if err != nil {
			break
		}
		valueType, err := readUint32(f)
		if err != nil {
			break
		}
		arch := meta.Architecture

		// Detect built-in vision encoder from keys like "{arch}.vision.block_count"
		if strings.Contains(key, ".vision.") {
			meta.HasVision = true
		}

		switch {
		case key == "general.alignment":
			// Padding between the tensor-info block and the data region.
			// Defaults to 32 when absent; only the last tensor's computed
			// size depends on it (see scanPLETensorBytes).
			if v, ok := readGGUFScalarInt(f, valueType); ok && v > 0 {
				alignment = int64(v)
			}
			continue

		case key == "general.architecture" && valueType == ggufTypeString:
			if v, err := readGGUFString(f); err == nil {
				meta.Architecture = v
			}
			continue

		case key == "general.base_model.0.repo_url" && valueType == ggufTypeString:
			if v, err := readGGUFString(f); err == nil {
				meta.BaseModelRepo = repoFromURL(v)
			}
			continue

		// general.sampling.* — author-recommended defaults. Out-of-range values
		// (same bounds the old scraper enforced) are dropped, but the value has
		// been consumed either way, so continue.
		case key == "general.sampling.temp":
			if v, ok := readGGUFScalarFloat(f, valueType); ok {
				if v >= 0 && v <= 4 {
					meta.SamplingTemp = &v
				}
				continue
			}
		case key == "general.sampling.top_p":
			if v, ok := readGGUFScalarFloat(f, valueType); ok {
				if v >= 0 && v <= 1 {
					meta.SamplingTopP = &v
				}
				continue
			}
		case key == "general.sampling.top_k":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				if v >= 1 && v <= 1000 {
					meta.SamplingTopK = &v
				}
				continue
			}
		case key == "general.sampling.min_p":
			if v, ok := readGGUFScalarFloat(f, valueType); ok {
				if v >= 0 && v <= 1 {
					meta.SamplingMinP = &v
				}
				continue
			}
		case key == "general.sampling.penalty_repeat":
			if v, ok := readGGUFScalarFloat(f, valueType); ok {
				if v > 0 && v <= 3 {
					meta.SamplingRepeatPenalty = &v
				}
				continue
			}

		case key == "tokenizer.chat_template" && valueType == ggufTypeString:
			if v, err := readGGUFString(f); err == nil {
				meta.SupportsTools = strings.Contains(v, "tools")
				meta.Reasoning = detectReasoning(v)
				meta.ReasoningChecked = true
			}
			continue

		// tokenizer.ggml.tokens — the vocab size IS the array length. Read
		// only the header (element type + count) and seek past the
		// potentially huge string contents.
		case key == "tokenizer.ggml.tokens" && valueType == ggufTypeArray:
			if elemType, count, ok := readGGUFArrayHeader(f); ok {
				meta.VocabSize = int(count)
				skipGGUFArrayBody(f, elemType, count)
			}
			continue

		case arch != "" && key == arch+".block_count":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				meta.NLayers = v
				continue
			}

		case arch != "" && key == arch+".embedding_length":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				meta.NEmbd = v
				continue
			}

		case arch != "" && key == arch+".attention.head_count":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				meta.NHead = v
				continue
			}

		case arch != "" && key == arch+".attention.head_count_kv":
			// Scalar for most models; a per-layer array for gemma-style models
			// whose GQA grouping differs between local and global layers.
			if valueType == ggufTypeArray {
				kvHeadCounts = readGGUFArrayInts(f)
				continue
			}
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				headCountKV = v
				meta.NKVHead = v
				continue
			}

		case arch != "" && key == arch+".context_length":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				meta.ContextLength = v
				continue
			}

		case arch != "" && key == arch+".attention.key_length":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				keyLen = v
				continue
			}
		case arch != "" && key == arch+".attention.value_length":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				valLen = v
				continue
			}
		case arch != "" && key == arch+".attention.key_length_swa":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				keyLenSWA = v
				continue
			}
		case arch != "" && key == arch+".attention.value_length_swa":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				valLenSWA = v
				continue
			}
		case arch != "" && key == arch+".attention.sliding_window":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				slidingWindow = v
				continue
			}
		case arch != "" && key == arch+".attention.indexer.key_length":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				meta.IndexerKeyLength = v
				continue
			}
		case arch != "" && key == arch+".full_attention_interval":
			if v, ok := readGGUFScalarInt(f, valueType); ok {
				fullAttnInterval = v
				continue
			}
		case arch != "" && key == arch+".attention.recurrent_layers":
			if valueType == ggufTypeArray {
				for _, x := range readGGUFArrayInts(f) {
					recurrentLayers = append(recurrentLayers, x != 0)
				}
				continue
			}
		case arch != "" && key == arch+".attention.sliding_window_pattern":
			// Per-layer bool array (gemma-3/4): true = local/sliding-window
			// layer. Other layouts (scalar interval) are left unmodeled, which
			// conservatively keeps those layers at full attention.
			if valueType == ggufTypeArray {
				for _, x := range readGGUFArrayInts(f) {
					swaPattern = append(swaPattern, x != 0)
				}
				continue
			}
		}

		// Skip values we don't capture (also handles type-mismatched reads above).
		skipGGUFValue(f, valueType)
	}

	computeKVScaling(meta, headCountKV, kvHeadCounts, keyLen, valLen, keyLenSWA, valLenSWA, slidingWindow, swaPattern,
		fullAttnInterval, recurrentLayers)

	// The KV loop consumed every key and either read or seek-skipped every
	// value, so the reader is now sitting on the tensor-info block. A file
	// truncated mid-header, or one whose tensor section we can't make sense
	// of, leaves PLEBytes at zero — the metadata gathered above is still
	// good, so a failure here must not fail the parse.
	meta.PLEBytes, meta.TokenEmbdBytes = scanPLETensorBytes(f, tensorCount, alignment)

	// A successful parse means reasoning was evaluated even if no chat template
	// was present — in which case Reasoning stays the unsupported zero value.
	// Normalize the empty toggle to the explicit "none" so the API contract
	// never emits a blank mechanism. Same for sampling: checked means the
	// general.sampling.* keys were looked for, present or not.
	meta.ReasoningChecked = true
	meta.SamplingChecked = true
	meta.PLEChecked = true
	meta.KVRecurrentChecked = true
	if meta.Reasoning.Toggle == "" {
		meta.Reasoning.Toggle = ReasoningToggleNone
	}

	return meta, nil
}

// isRecurrentLayer reports whether layer il holds recurrent state rather
// than a KV cache, following llama.cpp's own rule: an explicit
// attention.recurrent_layers array when the GGUF carries one, otherwise
// every layer except each full_attention_interval-th.
//
// Both are absent on a non-hybrid model, where every layer attends and
// this returns false throughout — so nothing changes for those.
func isRecurrentLayer(il, fullAttnInterval int, recurrentLayers []bool) bool {
	if il < len(recurrentLayers) {
		return recurrentLayers[il]
	}
	if fullAttnInterval > 0 {
		return (il+1)%fullAttnInterval != 0
	}
	return false
}

// computeKVScaling reduces the raw per-layer attention parameters into the
// compact KV-cache scaling factors stored on GGUFMeta. Falls back to uniform
// full attention with head_dim = n_embd/n_head when the richer keys are absent,
// which reproduces the legacy estimate exactly for non-gemma architectures.
func computeKVScaling(meta *GGUFMeta, headCountKV int, kvHeadCounts []int, keyLen, valLen, keyLenSWA, valLenSWA, slidingWindow int, swaPattern []bool,
	fullAttnInterval int, recurrentLayers []bool) {
	if meta.NLayers == 0 {
		return
	}

	embHeadDim := 0
	if meta.NHead > 0 {
		embHeadDim = meta.NEmbd / meta.NHead
	}
	kDim, vDim := keyLen, valLen
	if kDim == 0 {
		kDim = embHeadDim
	}
	if vDim == 0 {
		vDim = embHeadDim
	}
	kDimSWA, vDimSWA := keyLenSWA, valLenSWA
	if kDimSWA == 0 {
		kDimSWA = kDim
	}
	if vDimSWA == 0 {
		vDimSWA = vDim
	}
	if kDim+vDim == 0 {
		return // no head-dim info — leave KV factors zero, caller falls back
	}

	// Default per-layer KV head count: explicit scalar, else full attention.
	defaultKV := headCountKV
	if defaultKV == 0 {
		defaultKV = meta.NHead
	}

	full, swa, maxKV := 0, 0, meta.NKVHead
	for i := 0; i < meta.NLayers; i++ {
		// A recurrent layer carries linear-attention state instead of a KV
		// cache, so it contributes nothing here. Counting them made the
		// estimate for a hybrid too large by the full-attention interval —
		// four times over on both Qwen hybrids measured.
		if isRecurrentLayer(i, fullAttnInterval, recurrentLayers) {
			continue
		}
		meta.AttnLayers++
		kv := defaultKV
		if i < len(kvHeadCounts) {
			kv = kvHeadCounts[i]
		}
		if kv > maxKV {
			maxKV = kv
		}
		local := i < len(swaPattern) && swaPattern[i]
		if local && slidingWindow > 0 {
			swa += kv * (kDimSWA + vDimSWA)
		} else {
			full += kv * (kDim + vDim)
		}
	}
	meta.KVFullPerTok = full
	meta.KVSWAPerTok = swa
	if swa > 0 {
		meta.SlidingWindow = slidingWindow
	}
	// Representative scalar for display when head_count_kv was an array.
	if meta.NKVHead == 0 {
		meta.NKVHead = maxKV
	}
}

// GGUF value type constants
const (
	ggufTypeUint8   uint32 = 0
	ggufTypeInt8    uint32 = 1
	ggufTypeUint16  uint32 = 2
	ggufTypeInt16   uint32 = 3
	ggufTypeUint32  uint32 = 4
	ggufTypeInt32   uint32 = 5
	ggufTypeFloat32 uint32 = 6
	ggufTypeBool    uint32 = 7
	ggufTypeString  uint32 = 8
	ggufTypeArray   uint32 = 9
	ggufTypeUint64  uint32 = 10
	ggufTypeInt64   uint32 = 11
	ggufTypeFloat64 uint32 = 12
)

func readGGUFString(r io.ReadSeeker) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<20 { // 1MB sanity limit
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readUint32(r io.Reader) (uint32, error) {
	var v uint32
	err := binary.Read(r, binary.LittleEndian, &v)
	return v, err
}

// readGGUFScalarInt reads an integer-typed scalar value. It consumes bytes only
// when valueType is a recognized integer type; on a type mismatch it reads
// nothing and returns ok=false so the caller can fall back to skipGGUFValue
// (keeping the parser aligned).
func readGGUFScalarInt(r io.Reader, valueType uint32) (int, bool) {
	switch valueType {
	case ggufTypeUint8, ggufTypeInt8, ggufTypeBool:
		var v uint8
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return int(int8(v)), true
		}
	case ggufTypeUint16, ggufTypeInt16:
		var v uint16
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return int(int16(v)), true
		}
	case ggufTypeUint32, ggufTypeInt32:
		var v uint32
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return int(int32(v)), true
		}
	case ggufTypeUint64, ggufTypeInt64:
		var v uint64
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return int(int64(v)), true
		}
	}
	return 0, false
}

// readGGUFScalarFloat reads a float-typed scalar value, accepting integer
// types too (metadata authors sometimes write whole numbers as ints). Same
// consume-only-on-match contract as readGGUFScalarInt.
func readGGUFScalarFloat(r io.Reader, valueType uint32) (float64, bool) {
	switch valueType {
	case ggufTypeFloat32:
		var v float32
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return float64(v), true
		}
	case ggufTypeFloat64:
		var v float64
		if binary.Read(r, binary.LittleEndian, &v) == nil {
			return v, true
		}
	default:
		if v, ok := readGGUFScalarInt(r, valueType); ok {
			return float64(v), true
		}
	}
	return 0, false
}

// repoFromURL extracts "org/repo" from a huggingface.co URL, returning ""
// for anything that doesn't look like a HF repo URL.
func repoFromURL(u string) string {
	for _, prefix := range []string{"https://huggingface.co/", "http://huggingface.co/", "huggingface.co/"} {
		if rest, ok := strings.CutPrefix(u, prefix); ok {
			rest = strings.TrimSuffix(rest, "/")
			if parts := strings.Split(rest, "/"); len(parts) == 2 && parts[0] != "" && parts[1] != "" {
				return rest
			}
			return ""
		}
	}
	return ""
}

// readGGUFArrayInts reads an array value (its element type, count, and elements;
// the outer array type has already been consumed) and returns fixed-size
// numeric/bool elements as ints. It always consumes the full value so the parser
// stays aligned. Returns nil for unsupported element types.
func readGGUFArrayInts(r io.ReadSeeker) []int {
	elemType, err := readUint32(r)
	if err != nil {
		return nil
	}
	var count uint64
	if binary.Read(r, binary.LittleEndian, &count) != nil {
		return nil
	}
	if count > 1<<20 { // sanity bound
		return nil
	}
	out := make([]int, 0, count)
	for i := uint64(0); i < count; i++ {
		v, ok := readGGUFScalarInt(r, elemType)
		if !ok {
			// Unsupported element type — consume the remainder generically so
			// the stream stays aligned, then bail.
			remaining := int64(count - i)
			if sz := ggufFixedSize(elemType); sz > 0 {
				r.Seek(remaining*sz, io.SeekCurrent)
			}
			return out
		}
		out = append(out, v)
	}
	return out
}

// ggufFixedSize returns the byte size of a fixed-size GGUF value type, or 0 for variable types.
func ggufFixedSize(t uint32) int64 {
	switch t {
	case ggufTypeUint8, ggufTypeInt8, ggufTypeBool:
		return 1
	case ggufTypeUint16, ggufTypeInt16:
		return 2
	case ggufTypeUint32, ggufTypeInt32, ggufTypeFloat32:
		return 4
	case ggufTypeUint64, ggufTypeInt64, ggufTypeFloat64:
		return 8
	default:
		return 0
	}
}

// readGGUFArrayHeader reads an array value's element type and element
// count, leaving the reader positioned at the first element.
func readGGUFArrayHeader(r io.ReadSeeker) (elemType uint32, count uint64, ok bool) {
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return 0, 0, false
	}
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return 0, 0, false
	}
	return elemType, count, true
}

// skipGGUFArrayBody seeks past an array body whose element type and count
// have already been read. String arrays are stepped one length at a time;
// fixed-size element arrays are a single seek.
func skipGGUFArrayBody(r io.ReadSeeker, elemType uint32, count uint64) {
	if sz := ggufFixedSize(elemType); sz > 0 {
		r.Seek(int64(count)*sz, io.SeekCurrent)
		return
	}
	for i := uint64(0); i < count; i++ {
		var length uint64
		if binary.Read(r, binary.LittleEndian, &length) != nil {
			return
		}
		r.Seek(int64(length), io.SeekCurrent)
	}
}

func skipGGUFValue(r io.ReadSeeker, valueType uint32) {
	// Fixed-size types: seek past
	if sz := ggufFixedSize(valueType); sz > 0 {
		r.Seek(sz, io.SeekCurrent)
		return
	}

	switch valueType {
	case ggufTypeString:
		var length uint64
		if binary.Read(r, binary.LittleEndian, &length) != nil {
			return
		}
		r.Seek(int64(length), io.SeekCurrent)

	case ggufTypeArray:
		elemType, count, ok := readGGUFArrayHeader(r)
		if !ok {
			return
		}
		skipGGUFArrayBody(r, elemType, count)
	}
}

// pleTensorName is the tensor llama.cpp marks TENSOR_READ_LAZY: the
// per-layer / n-gram embedding table. The name is shared across every
// architecture that carries one (Gemma-3N, Gemma-4, Qwen4-Exp), because
// they all reuse the same LLM_TENSOR_PER_LAYER_TOKEN_EMBD entry in
// llama.cpp's tensor-name table.
//
// Matching on the tensor name rather than the architecture is deliberate:
// eligibility for on-demand reading is a property llama.cpp assigns in
// create_tensor(), and the set of architectures using it grows. A name
// match picks up the next one for free; an architecture allowlist would
// need editing every time.
const pleTensorName = "per_layer_token_embd.weight"

// tokenEmbdTensorName is the input embedding table. llama.cpp leaves it
// host-mapped rather than copying it to a device — measured on every
// model in the corpus, including ones with no per-layer table at all — so
// a VRAM estimate must not count it either. Its size is read from the
// tensor block rather than computed from vocab x n_embd, because
// publishers routinely quantize it differently from the body: both UD
// quants measured keep it at Q8_0 inside a Q4 model.
const tokenEmbdTensorName = "token_embd.weight"

// scanPLETensorBytes reads the tensor-info block and returns the size in
// bytes of the PLE table, or 0 if the file has no such tensor or the block
// can't be read. The reader must be positioned at the start of the block,
// which is where the metadata scan leaves it.
//
// Sizes come from the gaps between tensor data offsets rather than from
// each tensor's dimensions and quantization type. Both are correct, but
// the offsets need no table of ggml block sizes, so a GGUF quantized with
// a type this build has never heard of still measures correctly — and
// there is a steady supply of new ones. The cost is that a computed size
// includes any alignment padding that follows the tensor, at most
// alignment-1 bytes against a table measured in gigabytes.
func scanPLETensorBytes(f io.ReadSeeker, tensorCount uint64, alignment int64) (pleBytes, embBytes int64) {
	// A malformed count would otherwise have us loop for a very long time
	// on a file that isn't going to yield anything.
	const maxTensors = 1 << 20
	if tensorCount == 0 || tensorCount > maxTensors || alignment <= 0 {
		return 0, 0
	}

	var pleOffset int64 = -1
	var embOffset int64 = -1
	offsets := make([]int64, 0, tensorCount)

	for i := uint64(0); i < tensorCount; i++ {
		name, err := readGGUFString(f)
		if err != nil {
			return 0, 0
		}
		nDims, err := readUint32(f)
		if err != nil || nDims > 4 {
			return 0, 0
		}
		// Dimensions are read past; the offsets alone give us the size.
		if _, err := f.Seek(int64(nDims)*8, io.SeekCurrent); err != nil {
			return 0, 0
		}
		if _, err := readUint32(f); err != nil { // ggml type
			return 0, 0
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return 0, 0
		}
		offsets = append(offsets, int64(offset))
		if name == pleTensorName {
			pleOffset = int64(offset)
		}
		if name == tokenEmbdTensorName {
			embOffset = int64(offset)
		}
	}
	if pleOffset < 0 && embOffset < 0 {
		return 0, 0
	}

	// The data region starts after the tensor-info block, padded up to the
	// alignment; every offset above is relative to that point.
	infoEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, 0
	}
	dataStart := ((infoEnd + alignment - 1) / alignment) * alignment
	fileSize, err := f.Seek(0, io.SeekEnd)
	if err != nil || fileSize <= dataStart {
		return 0, 0
	}

	// The next offset after a tensor bounds it. Scanning for the smallest
	// offset greater than ours avoids assuming the tensors were written in
	// offset order. When nothing follows, the tensor runs to the end of
	// the file.
	sizeAt := func(off int64) int64 {
		if off < 0 {
			return 0
		}
		next := fileSize - dataStart
		for _, o := range offsets {
			if o > off && o < next {
				next = o
			}
		}
		if next <= off {
			return 0
		}
		return next - off
	}
	return sizeAt(pleOffset), sizeAt(embOffset)
}

// ggufShardPattern matches a split GGUF filename like
// "model-00001-of-00005.gguf". The naming is llama.cpp's, shared by every
// publisher that splits with its tooling.
var ggufShardPattern = regexp.MustCompile(`^(.+)-(\d{5})-of-(\d{5})\.gguf$`)

// ExpandShards returns every shard filename of a split GGUF, or a
// single-element slice for an unsplit one.
func ExpandShards(filename string) []string {
	m := ggufShardPattern.FindStringSubmatch(filename)
	if m == nil {
		return []string{filename}
	}
	base := m[1]
	total, _ := strconv.Atoi(m[3])
	shards := make([]string, total)
	for i := range total {
		shards[i] = fmt.Sprintf("%s-%05d-of-%05d.gguf", base, i+1, total)
	}
	return shards
}

// scanShardsForPLE looks for the per-layer embedding table in the sibling
// shards of a split GGUF.
//
// A split model's first shard — the one the registry records and the one
// callers hand to ParseGGUFMeta — holds all the metadata and, in the
// splits publishers actually ship, no tensors at all. So the architecture
// parses correctly from it while the tensor table, and with it the
// embedding table's size, sits in a later shard. Without this the
// per-layer embedding control never appears for exactly the models that
// have one, since a table big enough to matter belongs to a model big
// enough to be split.
//
// Returns 0 for an unsplit file, for siblings that aren't on disk (a
// partial download), or when no shard carries the table.
func scanShardsForTensors(path string) (pleBytes, embBytes int64) {
	dir, name := filepath.Split(path)
	shards := ExpandShards(name)
	if len(shards) < 2 {
		return 0, 0
	}
	for _, sh := range shards {
		if sh == name {
			continue // already parsed by the caller
		}
		f, err := os.Open(filepath.Join(dir, sh))
		if err != nil {
			continue
		}
		meta, err := ParseGGUFMetaFrom(f)
		f.Close()
		if err != nil {
			continue
		}
		if pleBytes == 0 {
			pleBytes = meta.PLEBytes
		}
		if embBytes == 0 {
			embBytes = meta.TokenEmbdBytes
		}
		if pleBytes > 0 && embBytes > 0 {
			break
		}
	}
	return pleBytes, embBytes
}
