package benchmark

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// SweepAxis is one parameter varied across a job's matrix. Field names
// a key in SweepFields; Values are the raw strings the user entered,
// parsed per the field's Kind when a cell is built.
//
// This is separate from ConfigOverrides — which fixes a value for every
// cell — so "run the whole job at ngl=99" stays a single entry rather
// than a one-element sweep.
type SweepAxis struct {
	Field  string   `json:"field"`
	Values []string `json:"values"`
}

// SweepKind describes how a value is parsed and rendered in the form.
type SweepKind string

const (
	SweepKindInt   SweepKind = "int"
	SweepKindBool  SweepKind = "bool"
	SweepKindStr   SweepKind = "string"
	SweepKindFloat SweepKind = "float"
)

// SweepChoice is one curated value offered in a parameter's dropdown.
type SweepChoice struct {
	Value string
	Label string
	// Params are the tunable settings this choice carries, rendered as
	// an expandable section under its checkbox. Only the speculative
	// decoding modes have them; editing one changes the submitted value
	// to the encoded form "mode:key=value,...". Mode is the bare mode
	// name the encoded Value starts with.
	Params []models.SpecModeParam
	Mode   string
}

// SweepField describes one sweepable parameter. The set function is the
// single place that knows how to turn a raw string into a
// ConfigOverrides mutation, so the form, the parser, and cell expansion
// cannot disagree about what a field means. Adding a field here makes it
// sweepable everywhere.
type SweepField struct {
	Name    string
	Label   string
	Kind    SweepKind
	Help    string
	Example string
	// RestartsRouter marks fields that only take effect when
	// llama-server reloads, which is what makes a sweep expensive.
	// Sampling params are sent per request and cost nothing.
	RestartsRouter bool
	// AffectsEval marks the fields that reach the capability-evaluation
	// command line, directly through MapConfigFlags (kv cache quant,
	// batch/ubatch, flash attention, gpu layers, threads, direct_io) or
	// through the placement resolution (gpu_assign and tensor_split,
	// which become --device/--tensor-split). A capability preset
	// collapses its cells over every OTHER axis: sweeping a parameter
	// the evaluation never sees would produce byte-identical
	// invocations under different labels — the mislabeled-results
	// failure this package guards against everywhere else.
	//
	// Deliberately NOT RestartsRouter: three load-time axes restart the
	// router yet never reach the eval — context_size (replaced by the
	// fixed evaluation context), spec_type (speculative decoding is
	// excluded from evaluations), and any future axis of that kind — so
	// keying the collapse on RestartsRouter would fan capability cells
	// out into identical runs.
	AffectsEval bool
	// Separator splits a value list. Exposed rather than hardcoded so the
	// form's live cell-count estimate splits exactly the way the parser
	// does — tensor_split values contain commas and so use "|".
	Separator string
	// Choices are the curated values offered as checkboxes. Selecting one
	// fixes the parameter; selecting several sweeps it. Empty means the
	// parameter has no meaningful enumeration.
	Choices []SweepChoice
	// FreeText renders a plain text input instead of a dropdown, for
	// parameters with no sensible curated values (a tensor split ratio,
	// a filesystem path). Sweeping still works via the separator.
	FreeText bool
	// AllowCustom adds a "Custom…" row to the dropdown so values outside
	// the curated list stay reachable without a code change. Off for
	// parameters that are genuinely closed sets, like booleans.
	AllowCustom bool
	// DynamicChoices names a choice set the API layer fills in at render
	// time because it depends on the host (e.g. how many GPUs exist).
	DynamicChoices string

	set func(o *ConfigOverrides, raw string) error
	// canon overrides the kind-based canonical form for fields whose
	// values have structure of their own (the encoded spec_type values).
	canon func(raw string) string
}

func parseInt(raw string) (int, error) {
	v, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0, fmt.Errorf("%q is not an integer", raw)
	}
	return v, nil
}

func parseFloat(raw string) (float64, error) {
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return 0, fmt.Errorf("%q is not a number", raw)
	}
	return v, nil
}

func parseBool(raw string) (bool, error) {
	v, err := strconv.ParseBool(strings.TrimSpace(raw))
	if err != nil {
		return false, fmt.Errorf("%q is not a boolean (use true/false)", raw)
	}
	return v, nil
}

func intField(name, label, help, example string, apply func(*ConfigOverrides, *int)) SweepField {
	return SweepField{
		Name: name, Label: label, Kind: SweepKindInt, Help: help,
		Example: example, RestartsRouter: true, Separator: ",",
		set: func(o *ConfigOverrides, raw string) error {
			v, err := parseInt(raw)
			if err != nil {
				return err
			}
			apply(o, &v)
			return nil
		},
	}
}

func strField(name, label, help, example string, apply func(*ConfigOverrides, *string)) SweepField {
	return SweepField{
		Name: name, Label: label, Kind: SweepKindStr, Help: help,
		Example: example, RestartsRouter: true, Separator: ",",
		set: func(o *ConfigOverrides, raw string) error {
			v := strings.TrimSpace(raw)
			apply(o, &v)
			return nil
		},
	}
}

func boolField(name, label, help string, apply func(*ConfigOverrides, *bool)) SweepField {
	return SweepField{
		Name: name, Label: label, Kind: SweepKindBool, Help: help,
		Example: "true,false", RestartsRouter: true, Separator: ",",
		set: func(o *ConfigOverrides, raw string) error {
			v, err := parseBool(raw)
			if err != nil {
				return err
			}
			apply(o, &v)
			return nil
		},
	}
}

// specValue is a parsed spec_type sweep value: a mode plus any
// parameter settings from the mode's expandable section.
type specValue struct {
	mode   string
	params map[string]string // key -> raw value, keys from models.SpecModeParams
}

// parseSpecValue parses the three accepted shapes: "none", a bare mode
// name, or "mode:key=value,key=value". Keys must belong to the mode
// (per models.SpecModeParams) and integer parameters must parse.
func parseSpecValue(raw string) (specValue, error) {
	v := strings.TrimSpace(raw)
	out := specValue{params: map[string]string{}}
	if v == "none" {
		return out, nil // mode "" = speculative decoding off
	}
	mode, rest, hasParams := strings.Cut(v, ":")
	mode = strings.TrimSpace(mode)
	allowed := models.SpecModeParams(mode)
	if len(allowed) == 0 {
		return out, fmt.Errorf("%q is not a speculative decoding mode", mode)
	}
	out.mode = mode
	if !hasParams {
		return out, nil
	}
	allowedKeys := map[string]bool{}
	for _, p := range allowed {
		allowedKeys[p.Key] = true
	}
	for _, pair := range strings.Split(rest, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		k, val, ok := strings.Cut(pair, "=")
		k, val = strings.TrimSpace(k), strings.TrimSpace(val)
		if !ok || k == "" {
			return out, fmt.Errorf("%q is not a key=value setting", pair)
		}
		if !allowedKeys[k] {
			return out, fmt.Errorf("%q is not a setting of the %s mode", k, mode)
		}
		if k == "draft_p_min" {
			if _, err := strconv.ParseFloat(val, 64); err != nil {
				return out, fmt.Errorf("draft_p_min %q is not a number", val)
			}
		} else {
			if _, err := strconv.Atoi(val); err != nil {
				return out, fmt.Errorf("%s %q is not an integer", k, val)
			}
		}
		out.params[k] = val
	}
	return out, nil
}

// applySpecValue is spec_type's set function: it writes the mode and any
// parameter settings onto the overrides. Parameters the value doesn't
// mention are left nil, so the model's saved values apply — the same
// inherit rule as every other parameter.
func applySpecValue(o *ConfigOverrides, raw string) error {
	sv, err := parseSpecValue(raw)
	if err != nil {
		return err
	}
	mode := sv.mode
	o.SpecType = &mode
	for k, val := range sv.params {
		switch k {
		case "draft_max":
			n, _ := strconv.Atoi(val)
			o.DraftMax = &n
		case "draft_min":
			n, _ := strconv.Atoi(val)
			o.DraftMin = &n
		case "draft_p_min":
			v := val
			o.DraftPMin = &v
		case "ngram_size_n":
			n, _ := strconv.Atoi(val)
			o.NgramSizeN = &n
		case "ngram_size_m":
			n, _ := strconv.Atoi(val)
			o.NgramSizeM = &n
		}
	}
	return nil
}

// encodeSpecValue renders a mode with its parameters' current values in
// the encoded form the parser accepts, skipping empty values (empty =
// inherit the model's saved setting). Bare mode when nothing is set.
func encodeSpecValue(mode string, params []models.SpecModeParam) string {
	var pairs []string
	for _, p := range params {
		if p.Default != "" {
			pairs = append(pairs, p.Key+"="+p.Default)
		}
	}
	if len(pairs) == 0 {
		return mode
	}
	return mode + ":" + strings.Join(pairs, ",")
}

// canonicalSpecValue renders a spec value in its parsed, sorted form so
// dedup catches values that differ only in spacing or parameter order.
// Unparseable values return trimmed raw; the parser rejects them later
// with a real error message.
func canonicalSpecValue(raw string) string {
	sv, err := parseSpecValue(raw)
	if err != nil {
		return strings.TrimSpace(raw)
	}
	if sv.mode == "" {
		return "none"
	}
	if len(sv.params) == 0 {
		return sv.mode
	}
	keys := make([]string, 0, len(sv.params))
	for k := range sv.params {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	pairs := make([]string, len(keys))
	for i, k := range keys {
		pairs[i] = k + "=" + sv.params[k]
	}
	return sv.mode + ":" + strings.Join(pairs, ",")
}

func floatField(name, label, help, example string, apply func(*ConfigOverrides, *float64)) SweepField {
	return SweepField{
		Name: name, Label: label, Kind: SweepKindFloat, Help: help,
		Example: example, RestartsRouter: false, Separator: ",",
		set: func(o *ConfigOverrides, raw string) error {
			v, err := parseFloat(raw)
			if err != nil {
				return err
			}
			apply(o, &v)
			return nil
		},
	}
}

// canonical returns the value in the form the parser resolves it to, so
// dedup catches values that differ as text but not as configuration —
// "1.0" vs "1", "8" vs "08". Raw text is returned for free-form kinds.
func (f SweepField) canonical(raw string) string {
	v := strings.TrimSpace(raw)
	if f.canon != nil {
		return f.canon(v)
	}
	switch f.Kind {
	case SweepKindInt:
		n, err := strconv.Atoi(v)
		if err != nil {
			return v
		}
		return strconv.Itoa(n)
	case SweepKindFloat:
		n, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return v
		}
		return strconv.FormatFloat(n, 'g', -1, 64)
	case SweepKindBool:
		b, err := strconv.ParseBool(v)
		if err != nil {
			return v
		}
		return strconv.FormatBool(b)
	default:
		return v
	}
}

// sweepFields is the registry. Keys match ConfigOverrides' JSON tags so
// a sweep axis and a fixed override name the same thing.
var sweepFields = map[string]SweepField{
	"gpu_layers": intField("gpu_layers", "GPU Layers", "Layers offloaded to GPU. 999 offloads everything.", "0,20,40,999",
		func(o *ConfigOverrides, v *int) { o.GPULayers = v }),
	// AffectsEval: the evaluation runs at its own fixed context, so the
	// model-config context size never reaches llama-perplexity.
	"context_size": intField("context_size", "Context Size", "Context window in tokens. Must fit the prompts the preset sends.", "4096,8192,32768",
		func(o *ConfigOverrides, v *int) { o.ContextSize = v }),
	"ubatch_size": intField("ubatch_size", "Micro-batch (-ub)", "Physical compute batch. The main prompt-processing knob; must be <= batch size. Blank leaves llama.cpp's default of 512.", "64,128,256,512,1024,2048",
		func(o *ConfigOverrides, v *int) { o.UBatchSize = v }),
	"batch_size": intField("batch_size", "Batch (-b)", "Logical batch size. Must be >= micro-batch. Blank leaves llama.cpp's default of 2048.", "1024,2048,4096",
		func(o *ConfigOverrides, v *int) { o.BatchSize = v }),
	"threads": intField("threads", "Threads", "CPU threads. Matters mainly when layers run on CPU.", "4,8,16",
		func(o *ConfigOverrides, v *int) { o.Threads = v }),
	"flash_attention": boolField("flash_attention", "Flash Attention", "Enables the flash-attention kernel. Required for some quantized KV cache types.",
		func(o *ConfigOverrides, v *bool) { o.FlashAttention = v }),
	"direct_io": boolField("direct_io", "Direct I/O", "Bypasses the page cache when loading weights.",
		func(o *ConfigOverrides, v *bool) { o.DirectIO = v }),
	// "auto" is llama.cpp's own default and is stored as the empty
	// string so the preset omits the flag entirely. It needs the
	// explicit spelling because a blank sweep value means "use the
	// model's saved setting", which is a different thing — the same
	// reason spec_type spells out "none".
	"ple_mode": {
		Name: "ple_mode", Label: "Per-Layer Embeddings", Kind: SweepKindStr,
		Help:    "How the per-layer embedding table is held. Auto follows llama.cpp and reads it on demand above 4 GiB. Only meaningful for models that carry such a table.",
		Example: "auto,on,off", RestartsRouter: true, Separator: ",",
		Choices: []SweepChoice{
			{Value: "auto", Label: "Auto (stream if over 4 GiB)"},
			{Value: "on", Label: "Stream from model file"},
			{Value: "off", Label: "Keep resident"},
		},
		set: func(o *ConfigOverrides, raw string) error {
			v := strings.TrimSpace(raw)
			switch v {
			case "auto":
				v = ""
			case "on", "off":
			default:
				return fmt.Errorf("%q is not a per-layer embedding mode (use auto, on or off)", raw)
			}
			o.PLEMode = &v
			return nil
		},
	},
	// Separator "|" rather than "," because flag values contain commas
	// of their own — an --override-tensor pattern list is the motivating
	// case, and splitting it on commas would shred one flag into several
	// bogus sweep points.
	"extra_flags": {
		Name: "extra_flags", Label: "Extra Flags", Kind: SweepKindStr,
		Help:           "Raw llama-server flags, appended verbatim to the preset. Separate sweep points with | since flag values contain spaces and commas.",
		Example:        "--load-mode none|--load-mode mmap",
		RestartsRouter: true, Separator: "|", FreeText: true,
		set: func(o *ConfigOverrides, raw string) error {
			v := strings.TrimSpace(raw)
			o.ExtraFlags = &v
			return nil
		},
	},
	"kv_cache_quant": strField("kv_cache_quant", "KV Cache Quant", "KV cache type, e.g. f16, q8_0, q4_0. Trades memory for accuracy.", "f16,q8_0",
		func(o *ConfigOverrides, v *string) { o.KVCacheQuant = v }),
	// gpu_assign and tensor_split reach the evaluation only through the
	// placement resolution (they become --device / --tensor-split via
	// models.GPUPlacementFlags), which is why they are AffectsEval.
	"gpu_assign": strField("gpu_assign", "GPU Assignment", "Which GPUs to use, e.g. all or 0,1.", "all,0",
		func(o *ConfigOverrides, v *string) { o.GPUAssign = v }),
	"tensor_split": strField("tensor_split", "Tensor Split", "Proportional split across GPUs, e.g. 1,1.", "1,1|3,1",
		func(o *ConfigOverrides, v *string) { o.TensorSplit = v }),
	// A value is either a bare mode name, "none" (an explicit
	// "speculative decoding off" — a blank value would be dropped as
	// "use the model's saved setting"), or a mode with parameter
	// settings in the encoded form "mode:draft_max=8,draft_p_min=0.9".
	// The form builds the encoded values from the expandable parameter
	// sections under each mode's checkbox; parseSpecValue is the single
	// parser for all three shapes.
	"spec_type": {
		Name: "spec_type", Label: "Speculative Decoding", Kind: SweepKindStr,
		Help:    "Speculative decoding mode. Select none to turn speculative decoding off for that cell; expand a mode to adjust its settings.",
		Example: "none,draft-mtp", RestartsRouter: true, Separator: ";",
		set:   applySpecValue,
		canon: canonicalSpecValue,
	},

	// Sampling params ride along with each request, so sweeping them
	// costs no router restarts.
	"temperature": floatField("temperature", "Temperature", "Sampling temperature.", "0,0.7,1.0",
		func(o *ConfigOverrides, v *float64) { o.Temperature = v }),
	"top_p": floatField("top_p", "Top P", "Nucleus sampling threshold.", "0.9,0.95",
		func(o *ConfigOverrides, v *float64) { o.TopP = v }),
	"min_p": floatField("min_p", "Min P", "Minimum probability threshold.", "0.0,0.05",
		func(o *ConfigOverrides, v *float64) { o.MinP = v }),
	"repeat_penalty": floatField("repeat_penalty", "Repeat Penalty", "Penalty applied to repeated tokens.", "1.0,1.1",
		func(o *ConfigOverrides, v *float64) { o.RepeatPenalty = v }),
	"top_k": intField("top_k", "Top K", "Limits sampling to the K most likely tokens.", "20,40",
		func(o *ConfigOverrides, v *int) { o.TopK = v }),
}

func init() {
	// AffectsEval classification: exactly the fields that reach the
	// capability-evaluation command line — the phase 01 direct mapping
	// (kv cache quant, batch/ubatch, flash attention, gpu layers,
	// threads, direct_io) plus the placement pair (gpu_assign,
	// tensor_split), which become --device / --tensor-split through
	// models.GPUPlacementFlags. The registry stays the single source: a
	// new mapped or excluded flag has exactly this one place to be
	// reflected. Everything else — the sampling params, context_size,
	// spec_type — keeps the zero value and collapses for capability
	// presets.
	for _, name := range []string{
		"gpu_layers", "ubatch_size", "batch_size", "threads",
		"flash_attention", "direct_io", "kv_cache_quant",
		"gpu_assign", "tensor_split",
	} {
		f := sweepFields[name]
		f.AffectsEval = true
		sweepFields[name] = f
	}

	// top_k is a sampling param despite being an int; intField defaults
	// to RestartsRouter, so correct it here rather than complicating the
	// constructor for a single case.
	f := sweepFields["top_k"]
	f.RestartsRouter = false
	sweepFields["top_k"] = f

	// tensor_split values are themselves comma-separated ("1,1"), so its
	// list needs a different separator.
	ts := sweepFields["tensor_split"]
	ts.Separator = "|"
	sweepFields["tensor_split"] = ts
}

// SweepFields returns the registry sorted by label, for rendering the
// job form.
func SweepFields() []SweepField {
	out := make([]SweepField, 0, len(sweepFields))
	for _, f := range sweepFields {
		out = append(out, f)
	}
	// Three tiers, then alphabetical within each. Parameters that reload
	// llama-server are the ones that move throughput, so they come first;
	// sampling settings don't and would otherwise scatter among them.
	// Free-text parameters sort last because they render as a different
	// control, and interleaving them looks like a rendering fault rather
	// than a distinction.
	tier := func(f SweepField) int {
		switch {
		case f.FreeText:
			return 2
		case !f.RestartsRouter:
			return 1
		default:
			return 0
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if ti, tj := tier(out[i]), tier(out[j]); ti != tj {
			return ti < tj
		}
		return out[i].Label < out[j].Label
	})
	return out
}

// LookupSweepField returns the field definition for name.
func LookupSweepField(name string) (SweepField, bool) {
	f, ok := sweepFields[name]
	return f, ok
}

// ValidateSweeps checks every axis names a known field and carries at
// least one valid value, and that no field is swept twice.
func ValidateSweeps(sweeps []SweepAxis) error {
	seen := map[string]bool{}
	for _, s := range sweeps {
		f, ok := sweepFields[s.Field]
		if !ok {
			return fmt.Errorf("unknown sweep field %q", s.Field)
		}
		if seen[s.Field] {
			return fmt.Errorf("%s is swept more than once", f.Label)
		}
		seen[s.Field] = true
		if len(s.Values) == 0 {
			return fmt.Errorf("%s: no values given", f.Label)
		}
		// Duplicates would expand into byte-identical cells: double the
		// runtime, and two rows showing one measurement as if it were
		// two data points. ParseSweepValues rejects them for form input;
		// this path is what JSON API callers hit.
		seenValue := map[string]bool{}
		for _, v := range s.Values {
			// Canonical form: "1.0" and "1" resolve to the same config
			// and would otherwise expand into byte-identical cells.
			c := f.canonical(v)
			if seenValue[c] {
				return fmt.Errorf("%s: duplicate value %q", f.Label, v)
			}
			seenValue[c] = true
			if err := f.set(&ConfigOverrides{}, v); err != nil {
				return fmt.Errorf("%s: %w", f.Label, err)
			}
		}
	}
	return nil
}

// CellOverrides merges a job's fixed overrides with one cell's swept
// values. The sweep wins where both set the same field, so a fixed value
// acts as the baseline for fields not being swept.
//
// Returns nil when neither is present, which callers use to skip the
// router restart entirely.
func CellOverrides(base *ConfigOverrides, values map[string]string) (*ConfigOverrides, error) {
	if base == nil && len(values) == 0 {
		return nil, nil
	}

	var merged ConfigOverrides
	if base != nil {
		merged = *base
	}
	// Apply in a stable order so an error message doesn't depend on map
	// iteration.
	names := make([]string, 0, len(values))
	for k := range values {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, name := range names {
		f, ok := sweepFields[name]
		if !ok {
			return nil, fmt.Errorf("unknown sweep field %q", name)
		}
		if err := f.set(&merged, values[name]); err != nil {
			return nil, fmt.Errorf("%s: %w", f.Label, err)
		}
	}
	return &merged, nil
}

// SweepRestartsRouter reports whether any axis forces a router reload,
// which is what makes a sweep slow. Used by the form to estimate cost.
func SweepRestartsRouter(sweeps []SweepAxis) bool {
	for _, s := range sweeps {
		if f, ok := sweepFields[s.Field]; ok && f.RestartsRouter {
			return true
		}
	}
	return false
}

func choices(pairs ...string) []SweepChoice {
	out := make([]SweepChoice, 0, len(pairs)/2)
	for i := 0; i+1 < len(pairs); i += 2 {
		out = append(out, SweepChoice{Value: pairs[i], Label: pairs[i+1]})
	}
	return out
}

// withChoices decorates a registry entry. Kept separate from the field
// constructors so the value-parsing logic and the presentation choices
// stay visibly distinct.
func withChoices(f SweepField, allowCustom bool, cs []SweepChoice) SweepField {
	f.Choices = cs
	f.AllowCustom = allowCustom
	return f
}

func init() {
	set := func(name string, allowCustom bool, cs []SweepChoice) {
		f := sweepFields[name]
		sweepFields[name] = withChoices(f, allowCustom, cs)
	}

	set("ubatch_size", true, choices(
		"64", "64", "128", "128", "256", "256",
		"512", "512 (default)", "1024", "1024", "2048", "2048",
	))
	set("batch_size", true, choices(
		"512", "512", "1024", "1024", "2048", "2048 (default)",
		"4096", "4096", "8192", "8192",
	))
	set("context_size", true, choices(
		"2048", "2K", "4096", "4K", "8192", "8K", "16384", "16K",
		"32768", "32K", "65536", "64K", "131072", "128K", "262144", "256K",
	))
	set("gpu_layers", true, choices(
		"0", "0 (CPU only)", "20", "20", "40", "40", "60", "60", "999", "999 (all)",
	))
	set("threads", true, choices(
		"4", "4", "8", "8", "16", "16", "32", "32",
	))
	set("flash_attention", false, choices("true", "on", "false", "off"))
	set("direct_io", false, choices("true", "on", "false", "off"))
	set("kv_cache_quant", true, choices(
		"f16", "f16 (no quant)", "q8_0", "q8_0", "q4_0", "q4_0",
	))
	// "none" (not an empty value, which would be dropped as unset) is the
	// explicit off entry; the field's set function maps it to "".
	set("spec_type", false, choices(
		"none", "Off (no speculative decoding)",
		"draft", "Draft Model", "draft-mtp", "MTP (self-speculation)",
		"ngram-simple", "N-gram Simple", "ngram-cache", "N-gram Cache",
		"ngram-map-k", "N-gram Map-K", "ngram-map-k4v", "N-gram Map-K4V",
		"ngram-mod", "N-gram Mod",
	))
	set("temperature", true, choices("0", "0 (greedy)", "0.7", "0.7", "1.0", "1.0"))
	set("top_p", true, choices("0.9", "0.9", "0.95", "0.95", "1.0", "1.0 (off)"))
	set("top_k", true, choices("20", "20", "40", "40", "0", "0 (disabled)"))
	set("min_p", true, choices("0", "0", "0.05", "0.05", "0.1", "0.1"))
	set("repeat_penalty", true, choices("1.0", "1.0 (off)", "1.05", "1.05", "1.1", "1.1"))

	// GPU assignment depends on how many GPUs this host has, so the API
	// layer fills it in at render time.
	ga := sweepFields["gpu_assign"]
	ga.DynamicChoices = "gpu_assign"
	ga.AllowCustom = true
	sweepFields["gpu_assign"] = ga

	// No sensible enumeration: a split ratio is free-form.
	ts2 := sweepFields["tensor_split"]
	ts2.FreeText = true
	sweepFields["tensor_split"] = ts2

	// Each speculative mode choice carries its tunable parameters (the
	// same set with the same recommended defaults as the model config
	// form), rendered as an expandable section under its checkbox. The
	// choice's submitted value is pre-encoded with those defaults —
	// selecting a mode in a job then behaves exactly like selecting it
	// on the model config form, and what the inputs show is what
	// submits. Editing an input re-encodes the value in the browser.
	st := sweepFields["spec_type"]
	for i := range st.Choices {
		mode := st.Choices[i].Value
		st.Choices[i].Params = models.SpecModeParams(mode)
		if len(st.Choices[i].Params) > 0 {
			st.Choices[i].Mode = mode
			st.Choices[i].Value = encodeSpecValue(mode, st.Choices[i].Params)
		}
	}
	sweepFields["spec_type"] = st
}

// SplitParams turns the unified "parameter → selected values" shape the
// job form posts into the two structures the runner uses: a single value
// fixes the parameter for every cell, two or more sweep it.
//
// The form deliberately has no separate override and sweep controls —
// they were two names for one question, and setting both left the
// override silently ignored.
func SplitParams(params map[string][]string) (*ConfigOverrides, []SweepAxis, error) {
	names := make([]string, 0, len(params))
	for k := range params {
		names = append(names, k)
	}
	sort.Strings(names)

	var overrides *ConfigOverrides
	var sweeps []SweepAxis

	for _, name := range names {
		f, ok := sweepFields[name]
		if !ok {
			return nil, nil, fmt.Errorf("unknown parameter %q", name)
		}

		// Drop blanks and duplicates; a blank means "use the model's
		// saved value", which is the absence of an entry.
		seen := map[string]bool{}
		var values []string
		for _, v := range params[name] {
			v = strings.TrimSpace(v)
			if v == "" {
				continue
			}
			c := f.canonical(v)
			if seen[c] {
				continue
			}
			seen[c] = true
			values = append(values, v)
		}
		if len(values) == 0 {
			continue
		}

		for _, v := range values {
			if err := f.set(&ConfigOverrides{}, v); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", f.Label, err)
			}
		}

		if len(values) == 1 {
			if overrides == nil {
				overrides = &ConfigOverrides{}
			}
			if err := f.set(overrides, values[0]); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", f.Label, err)
			}
			continue
		}
		sweeps = append(sweeps, SweepAxis{Field: name, Values: values})
	}
	return overrides, sweeps, nil
}

// MergeOverrides layers derived over base field by field, so overrides
// the caller supplied for fields the form cannot express survive an
// edit. A non-nil field in derived wins.
func MergeOverrides(base, derived *ConfigOverrides) *ConfigOverrides {
	if base == nil {
		return derived
	}
	if derived == nil {
		return base
	}
	out := *base
	d := reflect.ValueOf(*derived)
	o := reflect.ValueOf(&out).Elem()
	for i := 0; i < d.NumField(); i++ {
		if !d.Field(i).IsNil() {
			o.Field(i).Set(d.Field(i))
		}
	}
	return &out
}

// specParamTags are ConfigOverrides fields expressible through
// spec_type's encoded values rather than registry entries of their own.
var specParamTags = map[string]bool{
	"draft_max": true, "draft_min": true, "draft_p_min": true,
	"ngram_size_n": true, "ngram_size_m": true,
}

// IsSweepable reports whether a parameter has a form control, i.e.
// whether params can express it.
func IsSweepable(field string) bool {
	_, ok := sweepFields[field]
	return ok
}

// KeepUnsweepable returns a copy of o holding only fields the parameter
// registry cannot express. Used when merging a params-based edit over a
// caller-supplied override set: params own everything they can say, so
// leaving a parameter out of params means "unset it", not "keep it".
func KeepUnsweepable(o *ConfigOverrides) *ConfigOverrides {
	if o == nil {
		return nil
	}
	// JSON tags are the registry keys, so a field is sweepable exactly
	// when its tag names a registry entry — with one addition: the
	// speculative parameters have no registry entries of their own but
	// ARE expressible through spec_type's encoded values, so params own
	// them too and they must not carry through an edit.
	var out ConfigOverrides
	v := reflect.ValueOf(*o)
	t := v.Type()
	dst := reflect.ValueOf(&out).Elem()
	any := false
	for i := 0; i < v.NumField(); i++ {
		if v.Field(i).IsNil() {
			continue
		}
		tag := strings.Split(t.Field(i).Tag.Get("json"), ",")[0]
		if IsSweepable(tag) || specParamTags[tag] {
			continue
		}
		dst.Field(i).Set(v.Field(i))
		any = true
	}
	if !any {
		return nil
	}
	return &out
}

// ValidateSamplingSupport refuses a job that sets sampling parameters
// against a preset that cannot apply them.
//
// The llama-benchy path shells out with a fixed argument list and never
// sees RunConfig.Sampling, so a temperature sweep there would run three
// identical benchmarks and store them under three different labels —
// the mislabeled-results failure this package exists to avoid. Better to
// refuse than to silently measure the same thing repeatedly.
func ValidateSamplingSupport(presets []string, overrides *ConfigOverrides, sweeps []SweepAxis) error {
	var benchy []string
	for _, name := range presets {
		if GetPreset(name).EffectiveSource() == PresetSourceBenchy {
			benchy = append(benchy, name)
		}
	}
	if len(benchy) == 0 {
		return nil
	}

	var used []string
	seen := map[string]bool{}
	note := func(field string) {
		if f, ok := sweepFields[field]; ok && !f.RestartsRouter && !seen[field] {
			seen[field] = true
			used = append(used, f.Label)
		}
	}
	for _, sw := range sweeps {
		note(sw.Field)
	}
	if overrides != nil {
		v := reflect.ValueOf(*overrides)
		t := v.Type()
		for i := 0; i < v.NumField(); i++ {
			if v.Field(i).IsNil() {
				continue
			}
			note(strings.Split(t.Field(i).Tag.Get("json"), ",")[0])
		}
	}
	if len(used) == 0 {
		return nil
	}

	sort.Strings(used)
	return fmt.Errorf("%s cannot apply sampling settings (%s) — llama-benchy runs with a fixed argument list, so those cells would measure the same thing under different labels. Use an internal-* preset, or drop the sampling parameters",
		strings.Join(benchy, ", "), strings.Join(used, ", "))
}
