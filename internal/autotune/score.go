// Package autotune measures a model's speed under different settings and
// keeps the ones that win. It changes only settings that cannot change
// what the model answers — batch sizes, placement, speculative decoding —
// so a faster profile is faster at the same work, not at less of it.
package autotune

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"slices"
	"sort"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
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

// MeasurementPhrase is one goal's score in the words the screens use, such
// as "prompt 1400 tokens per second". Response scores are stored negated
// (higher is better for every goal), so the phrase flips the sign back.
func MeasurementPhrase(g Goal, s Score) string {
	switch g {
	case GoalGeneration:
		return fmt.Sprintf("generation %.1f tokens per second", s.Value)
	case GoalPrompt:
		return fmt.Sprintf("prompt %.0f tokens per second", s.Value)
	case GoalResponse:
		return fmt.Sprintf("a full answer in %.1f seconds", -s.Value)
	}
	return ""
}

// PartsPhrase says how a goal's score on a use case with several
// workloads is made up, such as "Average of two workloads: chat 112.1 →
// 129.5, coding 115.0 → 133.8 tokens per second". Without it a mixed
// result reads as one measurement, and a single workload's benchmark
// number looks like it disagrees. It gives the starting profile's numbers
// when they were measured, and is "" when there is nothing to break down.
func PartsPhrase(uc UseCase, g Goal, winner, baseline Candidate) string {
	shapes := Workloads[uc]
	parts := winner.Parts[g]
	if len(shapes) < 2 || len(parts) != len(shapes) {
		return ""
	}
	base := baseline.Parts[g]
	if len(base) != len(shapes) {
		base = nil
	}
	number := func(s Score) string {
		switch g {
		case GoalResponse:
			return fmt.Sprintf("%.1f", -s.Value)
		case GoalPrompt:
			return fmt.Sprintf("%.0f", s.Value)
		}
		return fmt.Sprintf("%.1f", s.Value)
	}
	items := make([]string, 0, len(shapes))
	for i, s := range shapes {
		item := s.Label + " "
		if base != nil {
			item += number(base[i]) + " → "
		}
		items = append(items, item+number(parts[i]))
	}
	kind, unit := "Average", "tokens per second"
	if g == GoalResponse {
		kind, unit = "Total", "seconds"
	}
	return fmt.Sprintf("%s of %s workloads: %s %s", kind, countWord(len(shapes)), strings.Join(items, ", "), unit)
}

// ClosestAlternative is the best setting other than the starting profile
// that the confirming stage measured for a goal, and how it compares with
// the profile: "3% slower". ok is false when there is none to compare —
// the profile was the only candidate with a measurement for the goal.
// When the starting profile wins, this is what tells the reader whether
// it won by a little or a lot.
func ClosestAlternative(rec *Autotune, g Goal) (alt Candidate, compared string, ok bool) {
	st, found := rec.Stage(StageConfirm)
	if !found {
		return Candidate{}, "", false
	}
	var base *Candidate
	for i, c := range st.Candidates {
		if len(c.Values) == 0 {
			base = &st.Candidates[i]
			continue
		}
		s, has := c.Scores[g]
		if !has {
			continue
		}
		if !ok || s.Value > alt.Scores[g].Value {
			alt, ok = c, true
		}
	}
	if !ok || base == nil {
		return Candidate{}, "", false
	}
	b, has := base.Scores[g]
	if !has {
		return Candidate{}, "", false
	}
	if g == GoalResponse {
		// Stored as negative seconds so that higher is better everywhere.
		return alt, percentFaster(-b.Value, -alt.Scores[g].Value, true), true
	}
	return alt, percentFaster(b.Value, alt.Scores[g].Value, false), true
}

// MarginPhrase explains why a faster alternative did not replace the
// starting profile: the gap between them is inside the margin of error
// the noise rule uses (the two measurements' spreads combined). Without
// it, "9% faster, but kept the profile" reads as a mistake. When either
// setting uses speculative decoding it also says why generation speed
// moves so much between runs.
func MarginPhrase(g Goal, base, alt Candidate) string {
	b, a := base.Scores[g], alt.Scores[g]
	gap := math.Abs(a.Value - b.Value)
	margin := math.Sqrt(a.Std*a.Std + b.Std*b.Std)
	format, unit, what := "%.1f", "tokens per second", "speed"
	switch g {
	case GoalPrompt:
		format = "%.0f"
	case GoalResponse:
		unit, what = "seconds", "time"
	}
	out := fmt.Sprintf("Your profile was kept because this is within the margin of error: "+
		"the gap is "+format+" %s, but the margin of error is ±"+format+" %s, "+
		"based on how much each setting's %s changed between repeated runs. "+
		"A gap this small may be chance, not a real gain",
		gap, unit, margin, unit, what)
	if g != GoalPrompt && (usesSpeculation(base.Config) || usesSpeculation(alt.Config)) {
		out += ". Speculative decoding makes generation speed change a lot between runs, " +
			"because how many of the guessed tokens are accepted depends on the text"
	}
	return out
}

func usesSpeculation(cfg benchmark.ConfigSnapshot) bool {
	return cfg.SpecType != "" || cfg.SpecAssist != ""
}

func countWord(n int) string {
	if n >= 0 && n < len(countWords) {
		return countWords[n]
	}
	return fmt.Sprint(n)
}

var countWords = []string{"zero", "one", "two", "three", "four"}

// Shape is one preset a use case measures with, and how its numbers are
// read: PromptTokens and OutputTokens are the request a response time is
// computed for, and PPSizeTokens is the prompt length prompt speed is
// read at (a longer one, where prompt speed is what the number is about).
// Label names the workload on the screens when a use case has several.
type Shape struct {
	Preset       string
	Label        string
	PromptTokens int
	OutputTokens int
	PPSizeTokens int
}

// Workloads maps a use case to the presets it measures with.
var Workloads = map[UseCase][]Shape{
	UseChat:  {chatShape},
	UseCode:  {codeShape},
	UseMixed: {chatShape, codeShape},
}

var (
	chatShape = Shape{Preset: "autotune-chat", Label: "chat", PromptTokens: 512, OutputTokens: 256, PPSizeTokens: 4096}
	codeShape = Shape{Preset: "autotune-code", Label: "coding", PromptTokens: 1536, OutputTokens: 512, PPSizeTokens: 1536}
)

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
	parts, err := ScoreParts(uc, g, runs)
	if err != nil {
		return Score{}, err
	}
	var values, vars []float64
	for _, p := range parts {
		values = append(values, p.Value)
		vars = append(vars, p.Std*p.Std)
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

// ScoreParts reads one goal out of each of a use case's workloads, in
// the order Workloads lists them. ScoreRuns combines them into the one
// score a goal is decided by; the screens show them apart, so a mixed
// result can be traced back to the workload that made it.
func ScoreParts(uc UseCase, g Goal, runs map[string]*benchmark.BenchmarkRun) ([]Score, error) {
	shapes := Workloads[uc]
	if len(shapes) == 0 {
		return nil, fmt.Errorf("unknown use case %q", uc)
	}
	out := make([]Score, 0, len(shapes))
	for _, s := range shapes {
		sc, err := scoreShape(s, g, runs[s.Preset])
		if err != nil {
			return nil, err
		}
		out = append(out, sc)
	}
	return out, nil
}

func scoreShape(s Shape, g Goal, run *benchmark.BenchmarkRun) (Score, error) {
	if run == nil || run.Summary == nil {
		return Score{}, fmt.Errorf("%w for preset %s", ErrNoMeasurement, s.Preset)
	}
	switch g {
	case GoalGeneration:
		row, ok := rowNearest(run, s.PromptTokens)
		if !ok || row.TGMean <= 0 {
			return Score{}, fmt.Errorf("%w: generation speed for preset %s", ErrNoMeasurement, s.Preset)
		}
		return Score{Value: row.TGMean, Std: row.TGStd}, nil
	case GoalPrompt:
		row, ok := rowNearest(run, s.PPSizeTokens)
		if !ok || row.PPMean <= 0 {
			return Score{}, fmt.Errorf("%w: prompt speed for preset %s", ErrNoMeasurement, s.Preset)
		}
		return Score{Value: row.PPMean, Std: row.PPStd}, nil
	case GoalResponse:
		row, ok := rowNearest(run, s.PromptTokens)
		if !ok || row.PPMean <= 0 || row.TGMean <= 0 {
			return Score{}, fmt.Errorf("%w: response time for preset %s", ErrNoMeasurement, s.Preset)
		}
		p, o := float64(s.PromptTokens), float64(s.OutputTokens)
		// The seconds depend on the speeds through 1/x, so the spread of
		// each speed carries over divided by its square.
		dp := p / (row.PPMean * row.PPMean) * row.PPStd
		do := o / (row.TGMean * row.TGMean) * row.TGStd
		return Score{Value: -(p/row.PPMean + o/row.TGMean), Std: math.Sqrt(dp*dp + do*do)}, nil
	}
	return Score{}, fmt.Errorf("unknown goal %q", g)
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
//
// The row's speeds are the median of its repetitions, not the mean: one
// repetition far from the others — a speculative decoding run that
// happened to guess nearly every token, say — would otherwise carry the
// whole result. The spread is still over every repetition, so a setting
// whose speed jumps around has to clear that spread in the noise rule.
func rowNearest(run *benchmark.BenchmarkRun, size int) (benchmark.SizeSummary, bool) {
	if row, ok := requestedRow(run, size); ok {
		return row, true
	}
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
	if !found {
		return best, false
	}
	var pp, tg []float64
	for _, r := range run.Results {
		if r.PromptTokens == best.PromptTokens {
			pp = append(pp, r.PromptTokPerSec)
			tg = append(tg, r.GenTokPerSec)
		}
	}
	if len(tg) > 0 {
		best.PPMean, best.TGMean = median(pp), median(tg)
	}
	return best, true
}

// requestedRow gathers the repetitions of the prompt size the run asked
// for that is closest to size, whatever llama.cpp counted for each. The
// per-size rows group by the counted number, and that differs between
// repetitions of one request: the first reads the whole prompt, the later
// ones reuse the cached start of it. A 512-token chat prompt counted 464
// tokens once and 422 twice, so the row nearest 512 held one repetition,
// with no spread at all. ok is false for a run that did not record what
// it asked for, which rowNearest then reads by its rows.
func requestedRow(run *benchmark.BenchmarkRun, size int) (benchmark.SizeSummary, bool) {
	if len(run.PromptTokens) == 0 || len(run.Results) == 0 {
		return benchmark.SizeSummary{}, false
	}
	nearest := func(n int) int {
		best := run.PromptTokens[0]
		for _, s := range run.PromptTokens[1:] {
			if abs(s-n) < abs(best-n) {
				best = s
			}
		}
		return best
	}
	target := nearest(size)
	row := benchmark.SizeSummary{PromptTokens: target}
	var pp, tg []float64
	var ttft float64
	for _, r := range run.Results {
		if nearest(r.PromptTokens) != target {
			continue
		}
		pp = append(pp, r.PromptTokPerSec)
		tg = append(tg, r.GenTokPerSec)
		ttft += r.TTFTMs
		row.GenTokens = r.GenTokens
	}
	if len(tg) == 0 {
		return benchmark.SizeSummary{}, false
	}
	row.Count = len(tg)
	row.PPMean, row.PPStd = median(pp), stdDev(pp)
	row.TGMean, row.TGStd = median(tg), stdDev(tg)
	row.AvgTTFTMs = ttft / float64(len(tg))
	return row, true
}

// stdDev is the population standard deviation, as the per-size rows
// report it.
func stdDev(xs []float64) float64 {
	var mean float64
	for _, x := range xs {
		mean += x
	}
	mean /= float64(len(xs))
	var sq float64
	for _, x := range xs {
		sq += (x - mean) * (x - mean)
	}
	return math.Sqrt(sq / float64(len(xs)))
}

// median is the middle value, or the mean of the middle two.
func median(xs []float64) float64 {
	s := slices.Clone(xs)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
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
	// Parts is each goal's score per workload, in Workloads order, when
	// the use case measures more than one — what Scores averages (speeds)
	// or adds up (response time). Runs saved before it existed lack it.
	Parts map[Goal][]Score `json:",omitempty"`
}

// Key identifies a candidate by its values, so the same settings from two
// stages are the same candidate.
func (c Candidate) Key() string {
	return models.JoinSorted(c.Values, " ")
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
	// Swapping the profile's draft method for an assist alone is a change
	// of mechanism too. Counting it as nothing made "ngram-mod instead of
	// MTP" read plainer than "MTP with ngram-mod" on a profile with MTP.
	// Turning speculation off altogether stays free: that is plainer.
	if cfg.SpecType != base.SpecType && (cfg.SpecType != "" || cfg.SpecAssist != "") {
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

// Pick chooses the winner for a goal: among the candidates no other
// candidate really beats, the plainest one, and of equally plain ones the
// one whose slow runs are fastest (its score less its spread). ranked is
// every candidate that has a score for the goal, fastest first, for
// choosing finalists.
//
// Every candidate is checked against every other, not only against the
// fastest: a setting whose speed jumps around can measure fastest with a
// spread so wide that a clearly slower setting falls inside it, and that
// slower setting must not win just for being plainer.
//
// The other way round, a setting must not survive on its own spread. One
// whose coding runs swung between 54 and 93 tokens per second sat inside
// the noise of every steady, faster setting, and then won for being one
// step plainer. So a candidate's spread counts in the comparison only as
// far as the faster setting's does (see outrun).
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

	// The fastest is never outrun, so there is always a winner.
	winnerComplexity := -1
	for _, c := range ranked {
		if outrunByAny(c, ranked, g) {
			continue // really slower than something
		}
		n := Complexity(c.Config, base)
		if winnerComplexity < 0 || n < winnerComplexity ||
			(n == winnerComplexity && slowRuns(c.Scores[g]) > slowRuns(winner.Scores[g])) {
			winner, winnerComplexity = c, n
		}
	}
	return winner, ranked, true
}

// outrun reports whether a is faster than b by more than the noise, for
// choosing between candidates. It is Beats with b's spread capped at a's:
// a steady setting is not outrun by a jumpy one that measured a little
// faster, but a jumpy setting cannot hide behind its own spread from a
// steady one that measured clearly faster.
func outrun(a, b Score) bool {
	bStd := math.Min(b.Std, a.Std)
	return a.Value-b.Value > math.Sqrt(a.Std*a.Std+bStd*bStd)
}

func outrunByAny(c Candidate, others []Candidate, g Goal) bool {
	for _, o := range others {
		if outrun(o.Scores[g], c.Scores[g]) {
			return true
		}
	}
	return false
}

// slowRuns is a score less its spread: roughly what the setting gives on
// its slower runs. Between equally plain settings it prefers the steady
// one when the two measured about the same, and the faster one when the
// gap is wider than the steady one's spread. Preferring the smallest
// spread outright picked 43.8 tokens per second over 62.6 because the
// slower setting jumped around a little less.
func slowRuns(s Score) float64 {
	return s.Value - s.Std
}

// Describe turns a candidate's sweep values into plain language for the
// screens. The empty set is the starting profile.
func Describe(values map[string]string) string {
	if len(values) == 0 {
		return "Starting profile, unchanged"
	}
	var parts []string
	for _, k := range slices.Sorted(maps.Keys(values)) {
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
	case "moe_cache_mib":
		if value == "0" {
			return "no expert cache"
		}
		return value + " MiB expert cache"
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
			extras = append(extras, "drafting with "+draftFileName(v))
		case benchmark.SpecDraftCacheKey:
			extras = append(extras, v+" draft cache")
		case "draft_sampling":
			extras = append(extras, v+" drafting")
		case benchmark.SpecDraftUBatchKey:
			extras = append(extras, "draft micro-batch "+v)
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

// draftFileName is the file a draft_model value names: a path, or a
// registry ID ("org--repo--file"). Cutting at the last "-" turned
// "…-DSpark-gguf--dspark-MiniCPM-V-4_7-35B-A3B-Q8_0" into just "Q8_0".
func draftFileName(v string) string {
	if i := strings.LastIndex(v, "/"); i >= 0 {
		v = v[i+1:]
	}
	if i := strings.LastIndex(v, "--"); i >= 0 {
		v = v[i+2:]
	}
	return v
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
		return "a DFlash or LiliCorr head"
	case "draft-dspark":
		return "a DSpark head"
	}
	return mode
}
