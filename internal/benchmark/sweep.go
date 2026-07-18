package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
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
	// Separator splits a value list. Exposed rather than hardcoded so the
	// form's live cell-count estimate splits exactly the way the parser
	// does — tensor_split values contain commas and so use "|".
	Separator string

	set func(o *ConfigOverrides, raw string) error
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

// sweepFields is the registry. Keys match ConfigOverrides' JSON tags so
// a sweep axis and a fixed override name the same thing.
var sweepFields = map[string]SweepField{
	"gpu_layers": intField("gpu_layers", "GPU Layers", "Layers offloaded to GPU. 999 offloads everything.", "0,20,40,999",
		func(o *ConfigOverrides, v *int) { o.GPULayers = v }),
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
	"kv_cache_quant": strField("kv_cache_quant", "KV Cache Quant", "KV cache type, e.g. f16, q8_0, q4_0. Trades memory for accuracy.", "f16,q8_0",
		func(o *ConfigOverrides, v *string) { o.KVCacheQuant = v }),
	"gpu_assign": strField("gpu_assign", "GPU Assignment", "Which GPUs to use, e.g. all or 0,1.", "all,0",
		func(o *ConfigOverrides, v *string) { o.GPUAssign = v }),
	"tensor_split": strField("tensor_split", "Tensor Split", "Proportional split across GPUs, e.g. 1,1.", "1,1|3,1",
		func(o *ConfigOverrides, v *string) { o.TensorSplit = v }),
	"spec_type": strField("spec_type", "Speculative Decoding", "Speculative decoding mode.", "none,draft-mtp",
		func(o *ConfigOverrides, v *string) { o.SpecType = v }),
	"draft_model_path": strField("draft_model_path", "Draft Model", "Path to the draft model GGUF.", "",
		func(o *ConfigOverrides, v *string) { o.DraftModelPath = v }),

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
	sort.Slice(out, func(i, j int) bool { return out[i].Label < out[j].Label })
	return out
}

// LookupSweepField returns the field definition for name.
func LookupSweepField(name string) (SweepField, bool) {
	f, ok := sweepFields[name]
	return f, ok
}

// ParseSweepValues splits a comma-separated list into trimmed, non-empty
// values, rejecting duplicates and validating each against the field's
// parser so a bad entry is caught when the job is defined rather than
// halfway through a long run.
//
// tensor_split uses "|" to separate values because its values contain
// commas.
func ParseSweepValues(field, raw string) ([]string, error) {
	f, ok := sweepFields[field]
	if !ok {
		return nil, fmt.Errorf("unknown sweep field %q", field)
	}

	sep := f.Separator
	if sep == "" {
		sep = ","
	}

	seen := map[string]bool{}
	var out []string
	for _, part := range strings.Split(raw, sep) {
		v := strings.TrimSpace(part)
		if v == "" {
			continue
		}
		if seen[v] {
			return nil, fmt.Errorf("%s: duplicate value %q", f.Label, v)
		}
		seen[v] = true
		if err := f.set(&ConfigOverrides{}, v); err != nil {
			return nil, fmt.Errorf("%s: %w", f.Label, err)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no values given", f.Label)
	}
	return out, nil
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
		for _, v := range s.Values {
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

// BuildSweeps turns raw "field → comma-separated list" form input into
// validated axes. Blank entries are skipped so an untouched form field
// doesn't create an empty axis. Keeping this server-side means the
// browser never has to reimplement value parsing.
func BuildSweeps(raw map[string]string) ([]SweepAxis, error) {
	names := make([]string, 0, len(raw))
	for k := range raw {
		names = append(names, k)
	}
	sort.Strings(names)

	var out []SweepAxis
	for _, name := range names {
		if strings.TrimSpace(raw[name]) == "" {
			continue
		}
		values, err := ParseSweepValues(name, raw[name])
		if err != nil {
			return nil, err
		}
		out = append(out, SweepAxis{Field: name, Values: values})
	}
	return out, nil
}
