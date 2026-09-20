// Package autotune measures a model's speed under different settings and
// keeps the ones that win. It changes only settings that cannot change
// what the model answers — batch sizes, placement, speculative decoding —
// so a faster profile is faster at the same work, not at less of it.
package autotune

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// UseCase is what the user says they use a model for. It decides the
// prompts autotune measures with, because speculative decoding in
// particular pays off very differently on text that repeats the prompt
// (editing code) and text that does not (open-ended chat).
type UseCase string

const (
	UseChat  UseCase = "chat"
	UseCode  UseCase = "code"
	UseMixed UseCase = "mixed"
)

// Goal is what "fastest" means for a run.
type Goal string

const (
	// GoalGeneration is tokens written per second.
	GoalGeneration Goal = "generation"
	// GoalPrompt is tokens of input read per second.
	GoalPrompt Goal = "prompt"
	// GoalResponse is the whole answer finishing soonest: reading the
	// prompt plus writing the answer.
	GoalResponse Goal = "response"
)

// Goals is the order the goals are reported in.
var Goals = []Goal{GoalGeneration, GoalPrompt, GoalResponse}

// GoalLabel is the goal in the words the screens use.
func GoalLabel(g Goal) string {
	switch g {
	case GoalGeneration:
		return "fastest generation"
	case GoalPrompt:
		return "fastest prompt"
	case GoalResponse:
		return "fastest response"
	}
	return string(g)
}

// Shape is one preset a use case measures with, and how its numbers are
// read: PromptTokens and OutputTokens are the request a response time is
// computed for, and PPSizeTokens is the prompt length prompt speed is
// read at (a longer one, where prompt speed is what the number is about).
type Shape struct {
	Preset       string
	PromptTokens int
	OutputTokens int
	PPSizeTokens int
}

// Workloads maps a use case to the presets it measures with.
var Workloads = map[UseCase][]Shape{
	UseChat: {{Preset: "autotune-chat", PromptTokens: 512, OutputTokens: 256, PPSizeTokens: 4096}},
	UseCode: {{Preset: "autotune-code", PromptTokens: 1536, OutputTokens: 512, PPSizeTokens: 1536}},
	UseMixed: {
		{Preset: "autotune-chat", PromptTokens: 512, OutputTokens: 256, PPSizeTokens: 4096},
		{Preset: "autotune-code", PromptTokens: 1536, OutputTokens: 512, PPSizeTokens: 1536},
	},
}

// UseCaseLabel is the use case in the words the screens use.
func UseCaseLabel(uc UseCase) string {
	switch uc {
	case UseChat:
		return "General chat"
	case UseCode:
		return "Coding and editing"
	case UseMixed:
		return "Mixed"
	}
	return string(uc)
}

// Presets returns the presets a use case needs measured.
func (uc UseCase) Presets() []string {
	var out []string
	for _, s := range Workloads[uc] {
		out = append(out, s.Preset)
	}
	return out
}

// Score is one measurement with its spread. Higher is always better:
// response time is carried as negative seconds, so every comparison in
// this package reads the same way.
type Score struct {
	Value float64
	Std   float64
}

// ErrNoMeasurement means a cell produced no usable numbers for a goal —
// it failed, or its run has no row at the size the goal reads.
var ErrNoMeasurement = errors.New("no measurement")

// ScoreRuns reads one goal out of a candidate's runs, keyed by preset.
func ScoreRuns(uc UseCase, g Goal, runs map[string]*benchmark.BenchmarkRun) (Score, error) {
	shapes := Workloads[uc]
	if len(shapes) == 0 {
		return Score{}, fmt.Errorf("unknown use case %q", uc)
	}

	var values, vars []float64
	for _, s := range shapes {
		run := runs[s.Preset]
		if run == nil || run.Summary == nil {
			return Score{}, fmt.Errorf("%w for preset %s", ErrNoMeasurement, s.Preset)
		}
		switch g {
		case GoalGeneration:
			row, ok := rowNearest(run, s.PromptTokens)
			if !ok || row.TGMean <= 0 {
				return Score{}, fmt.Errorf("%w: generation speed for preset %s", ErrNoMeasurement, s.Preset)
			}
			values = append(values, row.TGMean)
			vars = append(vars, row.TGStd*row.TGStd)
		case GoalPrompt:
			row, ok := rowNearest(run, s.PPSizeTokens)
			if !ok || row.PPMean <= 0 {
				return Score{}, fmt.Errorf("%w: prompt speed for preset %s", ErrNoMeasurement, s.Preset)
			}
			values = append(values, row.PPMean)
			vars = append(vars, row.PPStd*row.PPStd)
		case GoalResponse:
			row, ok := rowNearest(run, s.PromptTokens)
			if !ok || row.PPMean <= 0 || row.TGMean <= 0 {
				return Score{}, fmt.Errorf("%w: response time for preset %s", ErrNoMeasurement, s.Preset)
			}
			p, o := float64(s.PromptTokens), float64(s.OutputTokens)
			values = append(values, -(p/row.PPMean + o/row.TGMean))
			// The seconds depend on the speeds through 1/x, so the
			// spread of each speed carries over divided by its square.
			dp := p / (row.PPMean * row.PPMean) * row.PPStd
			do := o / (row.TGMean * row.TGMean) * row.TGStd
			vars = append(vars, dp*dp+do*do)
		default:
			return Score{}, fmt.Errorf("unknown goal %q", g)
		}
	}

	if g == GoalResponse {
		// Total time over the shapes, not an average: a mixed workload is
		// one of each.
		return Score{Value: sum(values), Std: math.Sqrt(sum(vars))}, nil
	}
	// A speed is averaged over the shapes, and its spread with it.
	n := float64(len(values))
	return Score{Value: sum(values) / n, Std: math.Sqrt(sum(vars)) / n}, nil
}

func sum(xs []float64) float64 {
	var t float64
	for _, x := range xs {
		t += x
	}
	return t
}

// rowNearest returns the per-size summary row closest to size tokens. A
// preset measures the sizes it was given, and llama.cpp reports what it
// actually tokenized, so the row is never exactly the asked-for number.
func rowNearest(run *benchmark.BenchmarkRun, size int) (benchmark.SizeSummary, bool) {
	rows := run.SizeRows()
	best, found := benchmark.SizeSummary{}, false
	for _, r := range rows {
		if r.Count == 0 {
			continue
		}
		if !found || abs(r.PromptTokens-size) < abs(best.PromptTokens-size) {
			best, found = r, true
		}
	}
	return best, found
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}

// Beats reports whether a is faster than b by more than the two
// measurements' own spread — the noise rule. Two runs of the same config
// differ a little; a difference inside that is not a result, and acting
// on it would leave a model carrying settings that bought nothing.
func Beats(a, b Score) bool {
	return a.Value-b.Value > math.Sqrt(a.Std*a.Std+b.Std*b.Std)
}

// Candidate is one set of settings autotune measured, as sweep values on
// top of the starting profile.
type Candidate struct {
	// Values are sweep field names to canonical values; empty is the
	// starting profile itself.
	Values map[string]string
	// Label is the setting in plain language, for the screens.
	Label string
	// Config is what the cell ran, recorded by its runs.
	Config benchmark.ConfigSnapshot
	// Scores is what it measured, by goal.
	Scores map[Goal]Score
	// Goals are the goals this candidate won, filled in by the runner.
	Goals []Goal
}

// Key identifies a candidate by its values, so the same settings from two
// stages are the same candidate.
func (c Candidate) Key() string {
	if len(c.Values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(c.Values))
	for k := range c.Values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s=%s", k, c.Values[k])
	}
	return b.String()
}

// Complexity counts how far a config is from the starting profile, so
// that among settings that measure the same, the plainest one wins.
// Speculative decoding costs the most: it is another mechanism to explain,
// and a draft method loads a second file.
func Complexity(cfg, base benchmark.ConfigSnapshot) int {
	n := 0
	if cfg.BatchSize != base.BatchSize {
		n++
	}
	if cfg.UBatchSize != base.UBatchSize {
		n++
	}
	if cfg.FlashAttention != base.FlashAttention {
		n++
	}
	if cfg.SplitMode != base.SplitMode {
		n++
	}
	if cfg.Threads != base.Threads {
		n++
	}
	if cfg.SpecAssist != base.SpecAssist && cfg.SpecAssist != "" {
		n += 2
	}
	if cfg.SpecType != base.SpecType && cfg.SpecType != "" {
		n += 3
	}
	for _, p := range []struct{ cur, base int }{
		{cfg.DraftMax, base.DraftMax}, {cfg.DraftMin, base.DraftMin},
		{cfg.AssistNMax, base.AssistNMax}, {cfg.AssistNMin, base.AssistNMin},
		{cfg.AssistNMatch, base.AssistNMatch}, {cfg.AssistSizeN, base.AssistSizeN},
		{cfg.AssistSizeM, base.AssistSizeM}, {cfg.AssistMinHits, base.AssistMinHits},
	} {
		if p.cur != p.base {
			n++
		}
	}
	return n
}

// Pick chooses the winner for a goal: the best measurement, unless a
// plainer candidate is within the noise of it, in which case the plainer
// one wins. ranked is every candidate that has a score for the goal,
// fastest first, for choosing finalists.
func Pick(cands []Candidate, g Goal, base benchmark.ConfigSnapshot) (winner Candidate, ranked []Candidate, ok bool) {
	for _, c := range cands {
		if _, has := c.Scores[g]; has {
			ranked = append(ranked, c)
		}
	}
	if len(ranked) == 0 {
		return Candidate{}, nil, false
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		return ranked[i].Scores[g].Value > ranked[j].Scores[g].Value
	})

	best := ranked[0]
	winner = best
	winnerComplexity := Complexity(best.Config, base)
	for _, c := range ranked[1:] {
		if Beats(best.Scores[g], c.Scores[g]) {
			continue // really slower
		}
		if n := Complexity(c.Config, base); n < winnerComplexity {
			winner, winnerComplexity = c, n
		}
	}
	return winner, ranked, true
}

// Describe turns a candidate's sweep values into plain language for the
// screens. The empty set is the starting profile.
func Describe(values map[string]string) string {
	if len(values) == 0 {
		return "Starting profile, unchanged"
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, DescribeValue(k, values[k]))
	}
	return strings.Join(parts, ", ")
}

// DescribeValue turns one sweep value into plain language.
func DescribeValue(field, value string) string {
	switch field {
	case "ubatch_size":
		return "prompt batch " + value
	case "batch_size":
		return "batch " + value
	case "flash_attention":
		if value == "true" {
			return "flash attention on"
		}
		return "flash attention off"
	case "split_mode":
		return value + " split"
	case "threads":
		return value + " threads"
	case "cpu_moe":
		return value + " expert layers on the CPU"
	case "spec_type":
		return describeSpec(value)
	}
	return field + " " + value
}

// describeSpec reads a speculative decoding value as a sentence:
// "draft-mtp+ngram-mod:draft_max=3" becomes "MTP with an ngram-mod
// assist, 3 draft tokens".
func describeSpec(value string) string {
	if value == "" || value == "none" {
		return "no speculative decoding"
	}
	modes, params, _ := strings.Cut(value, ":")
	var draft, assist string
	for _, m := range strings.Split(modes, "+") {
		if strings.HasPrefix(m, "ngram") {
			assist = m
		} else {
			draft = m
		}
	}
	var out string
	switch {
	case draft != "" && assist != "":
		out = draftLabel(draft) + " with an " + assist + " assist"
	case draft != "":
		out = draftLabel(draft)
	default:
		out = assist + " assist"
	}
	var extras []string
	for _, pair := range strings.Split(params, ",") {
		k, v, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		switch k {
		case "draft_max":
			extras = append(extras, v+" draft tokens")
		case benchmark.SpecDraftModelKey:
			if i := strings.LastIndexAny(v, "/-"); i >= 0 && i+1 < len(v) {
				v = v[i+1:]
			}
			extras = append(extras, "drafting with "+v)
		case "assist_n_max":
			extras = append(extras, "assist draft "+v)
		case "assist_size_n":
			extras = append(extras, "assist lookup "+v)
		}
	}
	if len(extras) > 0 {
		out += ", " + strings.Join(extras, ", ")
	}
	return out
}

func draftLabel(mode string) string {
	switch mode {
	case "draft-mtp":
		return "MTP draft layers"
	case "draft":
		return "a draft model"
	case "draft-eagle3":
		return "an EAGLE3 head"
	case "draft-dflash":
		return "a DFlash head"
	case "draft-dspark":
		return "a DSpark head"
	}
	return mode
}
