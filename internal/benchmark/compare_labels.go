package benchmark

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Telling compared runs apart.
//
// The compare view's bar charts labelled every bar with model, quant and
// build. In the case the view exists for — one model swept across a
// parameter grid — those three are identical for every run, so sixteen
// bars all read "Qwen3.5-9B-MTP (IQ4_NL) (vulkan)" and the winning bar
// is indistinguishable from the losing one. The information that
// separates them (batch size 1024, micro-batch 256) was recorded on the
// run and never shown.
//
// So the label is computed from the comparison rather than from the run:
// gather every dimension the runs could differ on, keep the ones that
// actually DO differ, and name the run by those. A sweep over
// micro-batch labels its bars by micro-batch. A comparison of two models
// at one setting labels its bars by model. A comparison where nothing
// varies falls back to the model name, because at that point the runs
// are repeats and there is nothing to choose between them.

// runDimension is one thing two runs can differ on: a display name and
// how to read the value off a run.
type runDimension struct {
	// Name is what the user sees, e.g. "micro-batch".
	Name string
	// Value returns this run's value, or "" when the run has none. An
	// empty value still counts as a value: a run with no KV quant
	// differs from one that has it.
	Value func(BenchmarkRun) string
	// IsSweep marks a dimension that came from a job's sweep rather than
	// from the run's config. Absence means something different for the
	// two: a run outside the sweep has no value for that parameter,
	// which is not worth printing, whereas a config field's empty state
	// is itself a setting.
	IsSweep bool
}

// comparisonDimensions returns every dimension worth telling runs apart
// by, in the order they should appear in a label: identity first
// (model, quant, build, preset), then the swept parameters, then the
// load-time config.
//
// Sweep dimensions come from the runs themselves rather than a fixed
// list, so a job that sweeps a parameter this code has never heard of
// still labels its bars correctly.
func comparisonDimensions(runs []BenchmarkRun) []runDimension {
	dims := []runDimension{
		{Name: "model", Value: func(r BenchmarkRun) string { return r.ModelName }},
		{Name: "quant", Value: func(r BenchmarkRun) string { return r.Quant }},
		{Name: "build", Value: func(r BenchmarkRun) string { return compareBuildLabel(r) }},
		{Name: "preset", Value: func(r BenchmarkRun) string { return r.Preset }},
	}

	// Swept parameters, sorted so the label reads the same way every
	// time rather than in Go's random map order.
	swept := map[string]bool{}
	for _, r := range runs {
		for k := range r.SweepValues {
			swept[k] = true
		}
	}
	names := make([]string, 0, len(swept))
	for k := range swept {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		name := name
		dims = append(dims, runDimension{
			Name:    sweepDisplayName(name),
			Value:   func(r BenchmarkRun) string { return r.SweepValues[name] },
			IsSweep: true,
		})
	}

	// Load-time config. A parameter that was swept is already covered
	// above; including it again would print it twice.
	cfg := []runDimension{
		{Name: "context", Value: func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.ContextSize) }},
		{Name: "batch", Value: func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.BatchSize) }},
		{Name: "micro-batch", Value: func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.UBatchSize) }},
		{Name: "KV cache", Value: func(r BenchmarkRun) string {
			if r.Config.KVCacheQuant == "" {
				return "f16"
			}
			return r.Config.KVCacheQuant
		}},
		{Name: "flash attention", Value: func(r BenchmarkRun) string { return onOff(r.Config.FlashAttention) }},
		{Name: "GPU layers", Value: func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.GPULayers) }},
		{Name: "GPUs", Value: func(r BenchmarkRun) string { return r.Config.GPUAssign }},
		{Name: "tensor split", Value: func(r BenchmarkRun) string { return r.Config.TensorSplit }},
		{Name: "threads", Value: func(r BenchmarkRun) string { return itoaOrEmpty(r.Config.Threads) }},
		{Name: "speculative decoding", Value: func(r BenchmarkRun) string { return r.Config.SpecType }},
	}
	for _, d := range cfg {
		if swept[configFieldForDimension(d.Name)] {
			continue
		}
		dims = append(dims, d)
	}
	return dims
}

// sweepDisplayName gives a swept parameter the same short name the
// config dimensions use, so a label never calls one thing
// "ubatch_size" in one place and "micro-batch" in another. Unknown
// parameters keep their own name, which is what the job form shows.
func sweepDisplayName(field string) string {
	switch field {
	case "context_size":
		return "context"
	case "batch_size":
		return "batch"
	case "ubatch_size":
		return "micro-batch"
	case "kv_cache_quant":
		return "KV cache"
	case "flash_attention":
		return "flash attention"
	case "gpu_layers":
		return "GPU layers"
	case "gpu_assign":
		return "GPUs"
	case "tensor_split":
		return "tensor split"
	case "spec_type":
		return "speculative decoding"
	}
	return field
}

// configFieldForDimension maps a config dimension's display name back to
// the sweep field name, so a swept parameter is not also listed as
// static config. Names not in the table are never swept.
func configFieldForDimension(name string) string {
	switch name {
	case "context":
		return "context_size"
	case "batch":
		return "batch_size"
	case "micro-batch":
		return "ubatch_size"
	case "KV cache":
		return "kv_cache_quant"
	case "flash attention":
		return "flash_attention"
	case "GPU layers":
		return "gpu_layers"
	case "GPUs":
		return "gpu_assign"
	case "tensor split":
		return "tensor_split"
	case "threads":
		return "threads"
	case "speculative decoding":
		return "spec_type"
	}
	return ""
}

func itoaOrEmpty(v int) string {
	if v == 0 {
		return ""
	}
	return strconv.Itoa(v)
}

func onOff(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

// compareBuildLabel renders a run's build the way the compare view does:
// profile and git ref when known, else the build id.
func compareBuildLabel(r BenchmarkRun) string {
	b := r.EffectiveBuild()
	parts := make([]string, 0, 2)
	if b.Profile != "" {
		parts = append(parts, b.Profile)
	}
	if b.GitRef != "" {
		parts = append(parts, b.GitRef)
	}
	if len(parts) == 0 && b.ID != "" {
		parts = append(parts, b.ID)
	}
	return strings.Join(parts, " · ")
}

// RunLabel is the compare view's presentation of one run.
type RunLabel struct {
	// Short names the run by what makes it different from the others.
	// It is the bar chart's visible label.
	Short string
	// Full lists everything known about the run, for the tooltip: the
	// identity, every swept value with its parameter name, the load-time
	// config, and the measured figures.
	Full string
}

// labelPriority orders the dimensions a short label draws on, most
// useful first. Swept parameters lead because a sweep comparison exists
// to choose between them; identity comes next; load-time config last,
// since it usually varies only incidentally when a heterogeneous set of
// runs is compared.
func labelPriority(d runDimension) int {
	if d.IsSweep {
		return 0
	}
	switch d.Name {
	case "model", "quant":
		return 1
	case "preset", "build":
		return 2
	}
	return 3
}

// shortLabelTarget is how many dimensions a bar label aims to carry.
// More than this and the label outgrows the chart; the rest stays in
// the tooltip. It is a target rather than a limit — uniqueness wins.
const shortLabelTarget = 4

// BuildRunLabels computes the compare view's label for each run, keyed
// by run ID.
//
// The short label uses as few dimensions as it can while still giving
// every run a DIFFERENT name. That guarantee is the point: a label that
// reads the same on the winning bar and the losing one answers nothing,
// which is what the view did before. It starts from the most useful
// handful and widens only if two runs would otherwise collide.
func BuildRunLabels(runs []BenchmarkRun) map[string]RunLabel {
	dims := comparisonDimensions(runs)

	// A dimension is worth naming only when the runs disagree about it.
	varying := make([]runDimension, 0, len(dims))
	for _, d := range dims {
		seen := map[string]bool{}
		for _, r := range runs {
			seen[d.Value(r)] = true
			if len(seen) > 1 {
				varying = append(varying, d)
				break
			}
		}
	}
	sort.SliceStable(varying, func(i, j int) bool {
		return labelPriority(varying[i]) < labelPriority(varying[j])
	})

	// Widen the label until every run has its own, starting from the
	// target width.
	n := shortLabelTarget
	if n > len(varying) {
		n = len(varying)
	}
	var shorts map[string]string
	for ; ; n++ {
		shorts = make(map[string]string, len(runs))
		used := make(map[string]bool, len(runs))
		unique := true
		for _, r := range runs {
			l := shortLabel(r, varying[:n])
			if used[l] {
				unique = false
			}
			used[l] = true
			shorts[r.ID] = l
		}
		if unique || n >= len(varying) {
			break
		}
	}

	out := make(map[string]RunLabel, len(runs))
	for _, r := range runs {
		out[r.ID] = RunLabel{Short: shorts[r.ID], Full: fullLabel(r, dims)}
	}
	return out
}

// shortLabel names a run by the given dimensions. When none are left —
// the compared runs differ in nothing this code can name — it falls
// back to the identity rather than rendering an empty bar label.
func shortLabel(r BenchmarkRun, dims []runDimension) string {
	parts := make([]string, 0, len(dims))
	for _, d := range dims {
		v := d.Value(r)
		if v == "" {
			// A run outside this sweep has no value for it. Printing
			// "batch —" for every such run is noise; leaving it out
			// says the same thing.
			if d.IsSweep {
				continue
			}
			v = "—"
		}
		// Identity dimensions read fine on their own; a bare "1024"
		// does not, so everything else carries its name.
		switch d.Name {
		case "model", "quant", "build", "preset":
			parts = append(parts, v)
		default:
			parts = append(parts, d.Name+" "+v)
		}
	}
	if len(parts) == 0 {
		return fallbackLabel(r)
	}
	return strings.Join(parts, " · ")
}

// fallbackLabel is the pre-existing model-quant-build label, used when
// the compared runs differ in nothing this code can name.
func fallbackLabel(r BenchmarkRun) string {
	label := r.ModelName
	if r.Quant != "" {
		label += " (" + r.Quant + ")"
	}
	if b := compareBuildLabel(r); b != "" {
		label += " (" + b + ")"
	}
	return label
}

// fullLabel lists every dimension with a value, plus the measured
// figures, as the tooltip text. Unlike Short it does not filter: the
// tooltip is where someone checks a setting they are not sure about, so
// it states the whole configuration even where the runs agree.
func fullLabel(r BenchmarkRun, dims []runDimension) string {
	var b strings.Builder
	for _, d := range dims {
		v := d.Value(r)
		if v == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%s: %s", d.Name, v)
	}
	if r.Summary != nil {
		fmt.Fprintf(&b, "\n\ngeneration: %.1f tok/s\nprompt: %.0f tok/s\ntime to first token: %.0f ms",
			r.Summary.AvgGenTokPerSec, r.Summary.AvgPromptTokPerSec, r.Summary.AvgTTFTMs)
	}
	if sizes := r.SizeRows(); len(sizes) > 0 {
		var labels []string
		for _, s := range sizes {
			if s.PromptTokens > 0 {
				labels = append(labels, "pp"+strconv.Itoa(s.PromptTokens))
			}
		}
		if len(labels) > 0 {
			fmt.Fprintf(&b, "\nprompt sizes measured: %s", strings.Join(labels, ", "))
		} else {
			b.WriteString("\nprompt sizes measured: averaged across sizes, not comparable to a single-size run")
		}
	}
	return b.String()
}

// PromptSizesText names the prompt lengths a run measured, for the
// compare view's details table.
//
// That table used to render one row per prompt length while the bar
// charts above it showed one bar per run, holding the run's average.
// Two rows for one bar, carrying different numbers, is the shape that
// made the table hard to read: nothing lined up between the two halves
// of the view. The table now shows one row per run with the same
// average the bar shows, and this column says what that average covers,
// so the reader can see at a glance whether two rows are measuring
// comparable things.
func (r BenchmarkRun) PromptSizesText() string {
	rows := r.SizeRows()
	if len(rows) == 0 {
		return ""
	}
	parts := make([]string, 0, len(rows))
	for _, s := range rows {
		if s.PromptTokens == 0 {
			// A legacy run, or one that recorded no per-length detail.
			// Its average is over an unknown set.
			return "not recorded"
		}
		parts = append(parts, strconv.Itoa(s.PromptTokens))
	}
	return strings.Join(parts, " / ")
}

// PromptSizesDetail breaks the average down by prompt length, for the
// tooltip on the cell PromptSizesText fills. The detail is worth
// keeping — prompt speed climbs steeply with prompt length, so an
// average over several lengths hides a wide spread — it just does not
// need a table row each.
func (r BenchmarkRun) PromptSizesDetail() string {
	rows := r.SizeRows()
	if len(rows) == 0 {
		return ""
	}
	if len(rows) == 1 && rows[0].PromptTokens > 0 {
		return fmt.Sprintf("Measured at %d-token prompts only, so this average is a single figure.", rows[0].PromptTokens)
	}
	if rows[0].PromptTokens == 0 {
		return "This run recorded no per-length detail, so its average is over an unknown mix of prompt lengths and cannot be compared with a run measured at one length."
	}

	var b strings.Builder
	fmt.Fprintf(&b, "This row averages %d prompt lengths. Prompt speed climbs steeply with prompt length, so the spread is wide:\n", len(rows))
	for _, s := range rows {
		fmt.Fprintf(&b, "\n%d tokens: %.0f prompt tok/s, %.1f generation tok/s, %.0f ms to first token",
			s.PromptTokens, s.PPMean, s.TGMean, s.AvgTTFTMs)
		if s.Count > 1 {
			fmt.Fprintf(&b, " (%d repetitions)", s.Count)
		}
	}
	return b.String()
}
