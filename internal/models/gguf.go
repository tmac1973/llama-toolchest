package models

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
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

	computeKVScaling(meta, headCountKV, kvHeadCounts, keyLen, valLen, keyLenSWA, valLenSWA, slidingWindow, swaPattern)

	// A successful parse means reasoning was evaluated even if no chat template
	// was present — in which case Reasoning stays the unsupported zero value.
	// Normalize the empty toggle to the explicit "none" so the API contract
	// never emits a blank mechanism. Same for sampling: checked means the
	// general.sampling.* keys were looked for, present or not.
	meta.ReasoningChecked = true
	meta.SamplingChecked = true
	if meta.Reasoning.Toggle == "" {
		meta.Reasoning.Toggle = ReasoningToggleNone
	}

	return meta, nil
}

// computeKVScaling reduces the raw per-layer attention parameters into the
// compact KV-cache scaling factors stored on GGUFMeta. Falls back to uniform
// full attention with head_dim = n_embd/n_head when the richer keys are absent,
// which reproduces the legacy estimate exactly for non-gemma architectures.
func computeKVScaling(meta *GGUFMeta, headCountKV int, kvHeadCounts []int, keyLen, valLen, keyLenSWA, valLenSWA, slidingWindow int, swaPattern []bool) {
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
