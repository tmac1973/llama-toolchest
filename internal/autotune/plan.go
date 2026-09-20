package autotune

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// PlanInput is what the planner needs to decide a stage's cells.
type PlanInput struct {
	Model *models.Model
	// Base is the starting profile's config: what every cell is measured
	// against, and what the values below are layered on.
	Base models.ModelConfig
	// Cards is how many GPUs the model is spread over, and Cores the
	// machine's logical cores. Both decide whether a setting is worth
	// measuring at all.
	Cards int
	Cores int
	// DraftCandidates lists the installed files usable by a draft
	// method, as the registry reports them.
	DraftCandidates func(mode string) []models.DraftCandidate
	// Finalists are the candidates earlier stages carried forward, by
	// stage key.
	Finalists map[string][]Candidate
}

// maxCells bounds a single stage, so a machine with many drafts or cores
// cannot turn one stage into an afternoon.
const maxCells = 40

// PlanStage returns the cells a stage measures: each one a set of sweep
// values layered on the starting profile. The first cell of the first
// stage is the profile itself, unchanged — the baseline every later
// comparison is against.
//
// skipped explains, in plain language, what could not be measured on this
// machine.
func PlanStage(key string, in PlanInput) (cells []map[string]string, axes []benchmark.SweepAxis, skipped []string) {
	switch key {
	case StageBatch:
		cells = planBatch(in)
	case StageSpec:
		cells, skipped = planSpec(in)
	case StageSpecParams:
		cells = planSpecParams(in)
	case StageConfirm:
		cells = planConfirm(in)
	}
	cells = dedupeCells(cells)
	if len(cells) > maxCells {
		cells = cells[:maxCells]
	}
	return cells, axesFor(cells), skipped
}

// planBatch measures how the prompt is fed to the model: the batch sizes,
// flash attention, and — only where they can matter — the split mode and
// the thread count.
func planBatch(in PlanInput) []map[string]string {
	ctx := in.Base.ContextSize
	if ctx == 0 && in.Model != nil {
		ctx = in.Model.ContextLength
	}
	type batchPair struct{ batch, ubatch int }
	pairs := []batchPair{{2048, 256}, {2048, 512}, {2048, 1024}, {2048, 2048}, {4096, 4096}}

	var base []map[string]string
	for _, p := range pairs {
		if ctx > 0 && p.ubatch > ctx {
			// A micro-batch larger than the context is never used whole.
			continue
		}
		base = append(base, map[string]string{
			"batch_size":  strconv.Itoa(p.batch),
			"ubatch_size": strconv.Itoa(p.ubatch),
		})
	}

	// Flash attention off is only worth measuring where llama.cpp would
	// load it: it refuses a tensor split or a quantized KV cache without.
	cells := crossBool(base, "flash_attention", flashOptions(in.Base))
	if in.Cards > 1 {
		cells = crossValues(cells, "split_mode", []string{"layer", "tensor"})
	}
	if onCPU(in) && in.Cores >= 4 {
		cells = crossValues(cells, "threads", threadOptions(in.Cores))
	}
	// The starting profile itself, first: every later stage compares
	// against it.
	return append([]map[string]string{{}}, cells...)
}

// flashOptions is the flash-attention values worth measuring: both, or
// only the one the profile can run.
func flashOptions(base models.ModelConfig) []bool {
	off := base
	off.FlashAttention = false
	if off.ValidateFlashAttention() != nil {
		return []bool{true}
	}
	return []bool{true, false}
}

// onCPU reports whether part of the model runs on the CPU, which is the
// only case where the thread count changes anything worth measuring.
func onCPU(in PlanInput) bool {
	if in.Base.CPUMoE > 0 {
		return true
	}
	return in.Model != nil && in.Model.NLayers > 0 && in.Base.GPULayers > 0 && in.Base.GPULayers < in.Model.NLayers
}

// threadOptions offers a quarter, a half and three quarters of the
// machine's logical cores, deduplicated and at least one.
func threadOptions(cores int) []string {
	seen := map[int]bool{}
	var out []string
	for _, n := range []int{cores / 4, cores / 2, cores * 3 / 4} {
		if n < 1 {
			n = 1
		}
		if seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, strconv.Itoa(n))
	}
	return out
}

// planSpec measures speculative decoding: each draft method this machine
// can serve, each n-gram assist, and every pair of the two. The pairs are
// the point — a draft method and an assist draft from different things,
// and together they often beat either alone.
func planSpec(in PlanInput) ([]map[string]string, []string) {
	drafts, skipped := draftOptions(in)
	assists := []string{}
	for _, m := range models.AssistModes() {
		assists = append(assists, m.Name)
	}

	values := []string{benchmark.EncodeSpecValue("", "", nil)} // "none"
	for _, d := range drafts {
		values = append(values, benchmark.EncodeSpecValue(d.mode, "", d.params))
	}
	for _, a := range assists {
		values = append(values, benchmark.EncodeSpecValue("", a, defaultParams(nil, models.SpecAssistParams(a))))
	}
	for _, d := range drafts {
		for _, a := range assists {
			params := map[string]string{}
			for k, v := range d.params {
				params[k] = v
			}
			for k, v := range defaultParams(nil, models.SpecAssistParams(a)) {
				params[k] = v
			}
			values = append(values, benchmark.EncodeSpecValue(d.mode, a, params))
		}
	}

	var cells []map[string]string
	for _, carried := range carryForward(in, StageBatch) {
		for _, v := range values {
			cells = append(cells, withValue(carried, "spec_type", v))
		}
	}
	return cells, skipped
}

// draftOption is one draft method this machine can run, with the settings
// that name its file.
type draftOption struct {
	mode   string
	params map[string]string
}

// draftOptions lists the draft methods that can actually run here, and
// says why the others cannot.
func draftOptions(in PlanInput) ([]draftOption, []string) {
	var out []draftOption
	var skipped []string
	m := in.Model

	// Built-in MTP layers, or a head already attached to the profile.
	switch {
	case m != nil && m.NextNLayers > 0:
		out = append(out, draftOption{"draft-mtp", defaultParams(nil, models.SpecDraftParams("draft-mtp"))})
	case in.Base.MtpPath != "":
		out = append(out, draftOption{"draft-mtp", defaultParams(
			map[string]string{benchmark.SpecDraftModelKey: in.Base.MtpPath},
			models.SpecDraftParams("draft-mtp"))})
	default:
		skipped = append(skipped, "MTP: this model has no draft layers of its own and no MTP head is installed")
	}

	for _, mode := range []string{"draft", "draft-eagle3", "draft-dflash", "draft-dspark"} {
		var cands []models.DraftCandidate
		if in.DraftCandidates != nil {
			cands = in.DraftCandidates(mode)
		}
		if len(cands) == 0 {
			skipped = append(skipped, fmt.Sprintf("%s: no file for it is installed", draftName(mode)))
			continue
		}
		// The smallest candidates first: a drafter is worth having only
		// while it is much cheaper than the model it drafts for.
		sort.SliceStable(cands, func(i, j int) bool { return cands[i].SizeGB < cands[j].SizeGB })
		limit := 2
		if mode != "draft" {
			limit = 1 // a converted head is specific to the model; there is one
		}
		for i, c := range cands {
			if i >= limit {
				break
			}
			out = append(out, draftOption{mode, defaultParams(
				map[string]string{benchmark.SpecDraftModelKey: c.ID},
				models.SpecDraftParams(mode))})
		}
	}
	return out, skipped
}

func draftName(mode string) string {
	switch mode {
	case "draft":
		return "Draft model"
	case "draft-eagle3":
		return "EAGLE3"
	case "draft-dflash":
		return "DFlash"
	case "draft-dspark":
		return "DSpark"
	}
	return mode
}

// defaultParams fills a spec value's settings with the recommended
// defaults for its mode, keeping anything already set.
func defaultParams(into map[string]string, params []models.SpecModeParam) map[string]string {
	out := map[string]string{}
	for k, v := range into {
		out[k] = v
	}
	for _, p := range params {
		if p.Default == "" {
			continue
		}
		if _, ok := out[p.Key]; !ok {
			out[p.Key] = p.Default
		}
	}
	return out
}

// planSpecParams tunes the settings of whatever speculative decoding the
// previous stage chose: how many tokens are drafted, and how far the
// n-gram assist looks back.
func planSpecParams(in PlanInput) []map[string]string {
	var cells []map[string]string
	for _, c := range carryForward(in, StageSpec) {
		spec := c["spec_type"]
		if spec == "" || spec == "none" {
			continue // nothing to tune
		}
		mode, assist, params := splitSpec(spec)
		// Only a draft method has a draft length; an n-gram assist on its
		// own would be handed a setting it does not have, and the cell
		// would fail rather than measure anything.
		draftValues := []string{""}
		if mode != "" {
			draftValues = paramValues(params, "draft_max", models.SpecDraftParams(mode), 3)
		}
		assistKey := assistParamKey(assist)
		assistValues := paramValues(params, assistKey, models.SpecAssistParams(assist), 0)

		const perFinalist = 12
		added := 0
		for _, d := range draftValues {
			for _, a := range assistValues {
				if added >= perFinalist {
					break
				}
				next := map[string]string{}
				for k, v := range params {
					next[k] = v
				}
				if d != "" {
					next["draft_max"] = d
				}
				if a != "" && assistKey != "" {
					next[assistKey] = a
				}
				value := benchmark.EncodeSpecValue(mode, assist, next)
				if value == spec {
					continue // already measured as the finalist itself
				}
				cells = append(cells, withValue(c, "spec_type", value))
				added++
			}
		}
	}
	return cells
}

// assistParamKey is the setting worth tuning for an n-gram assist: how
// many tokens it drafts (ngram-mod) or how far back it looks (the rest).
// ngram-cache has no settings.
func assistParamKey(assist string) string {
	switch assist {
	case "":
		return ""
	case "ngram-mod":
		return "assist_n_max"
	case "ngram-cache":
		return ""
	}
	return "assist_size_n"
}

// paramValues returns the values to try for one setting: half, the
// current value, half again and double. An empty list means "leave it",
// which keeps the finalist's own value.
func paramValues(params map[string]string, key string, table []models.SpecModeParam, fallback int) []string {
	if key == "" {
		return []string{""}
	}
	cur := 0
	if v, ok := params[key]; ok {
		cur, _ = strconv.Atoi(v)
	}
	if cur == 0 {
		for _, p := range table {
			if p.Key == key && p.Default != "" {
				cur, _ = strconv.Atoi(p.Default)
			}
		}
	}
	if cur == 0 {
		cur = fallback
	}
	if cur == 0 {
		return []string{""}
	}
	seen := map[int]bool{cur: true}
	out := []string{strconv.Itoa(cur)}
	for _, n := range []int{cur / 2, cur * 3 / 2, cur * 2} {
		if n < 1 || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, strconv.Itoa(n))
	}
	return out
}

// splitSpec takes a spec value apart into its modes and settings.
func splitSpec(value string) (mode, assist string, params map[string]string) {
	params = map[string]string{}
	modes, rest, _ := strings.Cut(value, ":")
	for _, m := range strings.Split(modes, "+") {
		switch {
		case m == "" || m == "none":
		case strings.HasPrefix(m, "ngram"):
			assist = m
		default:
			mode = m
		}
	}
	for _, pair := range strings.Split(rest, ",") {
		if k, v, ok := strings.Cut(pair, "="); ok {
			params[k] = v
		}
	}
	return mode, assist, params
}

// planConfirm measures the finalists side by side, at the end, so the
// numbers that decide the winner were all taken in the same conditions:
// the best batch settings against the best speculative settings, plus the
// starting profile.
func planConfirm(in PlanInput) []map[string]string {
	batch := topByResponse(in.Finalists[StageBatch], 2)
	spec := distinct(append(append([]Candidate{}, in.Finalists[StageSpecParams]...), in.Finalists[StageSpec]...), 3)

	cells := []map[string]string{{}} // the starting profile
	for _, b := range batch {
		for _, s := range spec {
			merged := map[string]string{}
			for k, v := range b.Values {
				merged[k] = v
			}
			for k, v := range s.Values {
				merged[k] = v
			}
			cells = append(cells, merged)
		}
		if len(spec) == 0 {
			cells = append(cells, copyValues(b.Values))
		}
	}
	return cells
}

// topByResponse returns the n candidates with the best response time.
func topByResponse(cands []Candidate, n int) []Candidate {
	ranked := append([]Candidate{}, cands...)
	sort.SliceStable(ranked, func(i, j int) bool {
		return ranked[i].Scores[GoalResponse].Value > ranked[j].Scores[GoalResponse].Value
	})
	if len(ranked) > n {
		ranked = ranked[:n]
	}
	return ranked
}

// distinct keeps at most n candidates with different settings.
func distinct(cands []Candidate, n int) []Candidate {
	seen := map[string]bool{}
	var out []Candidate
	for _, c := range cands {
		k := c.Key()
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, c)
		if len(out) >= n {
			break
		}
	}
	return out
}

// carryForward returns the settings a stage builds on: the finalists of
// an earlier stage, or the starting profile alone before it has run.
func carryForward(in PlanInput, stage string) []map[string]string {
	var out []map[string]string
	for _, c := range distinct(in.Finalists[stage], 3) {
		out = append(out, copyValues(c.Values))
	}
	if len(out) == 0 {
		out = []map[string]string{{}}
	}
	return out
}

func copyValues(v map[string]string) map[string]string {
	out := make(map[string]string, len(v))
	for k, val := range v {
		out[k] = val
	}
	return out
}

func withValue(base map[string]string, key, value string) map[string]string {
	out := copyValues(base)
	out[key] = value
	return out
}

func crossValues(cells []map[string]string, key string, values []string) []map[string]string {
	var out []map[string]string
	for _, c := range cells {
		for _, v := range values {
			out = append(out, withValue(c, key, v))
		}
	}
	return out
}

func crossBool(cells []map[string]string, key string, values []bool) []map[string]string {
	strs := make([]string, len(values))
	for i, v := range values {
		strs[i] = strconv.FormatBool(v)
	}
	return crossValues(cells, key, strs)
}

// dedupeCells drops repeated settings, which the crosses above can
// produce when a finalist already carries one of them.
func dedupeCells(cells []map[string]string) []map[string]string {
	seen := map[string]bool{}
	var out []map[string]string
	for _, c := range cells {
		k := Candidate{Values: c}.Key()
		if seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, c)
	}
	return out
}

// axesFor lists the fields a stage varies, for the job detail view. The
// values are every value that appears, so the view can label the cells.
func axesFor(cells []map[string]string) []benchmark.SweepAxis {
	values := map[string][]string{}
	seen := map[string]bool{}
	var order []string
	for _, c := range cells {
		for k, v := range c {
			if !seen[k+"\x00"+v] {
				seen[k+"\x00"+v] = true
				if _, ok := values[k]; !ok {
					order = append(order, k)
				}
				values[k] = append(values[k], v)
			}
		}
	}
	sort.Strings(order)
	out := make([]benchmark.SweepAxis, 0, len(order))
	for _, k := range order {
		out = append(out, benchmark.SweepAxis{Field: k, Values: values[k]})
	}
	return out
}

// StageJobInput is what BuildStageJob needs beyond the record.
type StageJobInput struct {
	// Profile is the starting profile, as the job will measure it.
	Profile benchmark.BaseProfile
	// Cells and Axes come from PlanStage.
	Cells []map[string]string
	Axes  []benchmark.SweepAxis
	// ModelName is for the job's name; JobID is the ID to give it.
	ModelName string
	JobID     string
}

// BuildStageJob turns a stage's plan into an ordinary benchmark job: one
// cell per (settings × preset), measured against the starting profile.
// Nothing about it is special to autotune except the two fields that link
// it back, so it appears in the benchmark history like any other job.
func BuildStageJob(rec *Autotune, stageKey string, in StageJobInput) benchmark.BenchmarkJob {
	presets := rec.UseCase.Presets()
	stageNo := 1
	for i, s := range StageOrder {
		if s.Key == stageKey {
			stageNo = i + 1
		}
	}
	name := in.ModelName
	if name == "" {
		name = rec.ModelID
	}
	profile := in.Profile
	job := benchmark.BenchmarkJob{
		ID:            in.JobID,
		Name:          fmt.Sprintf("Autotune: %s — stage %d of %d, %s", name, stageNo, len(StageOrder), StageTitle(stageKey)),
		Description:   fmt.Sprintf("Measuring from the %q profile for %s.", rec.BaseProfile, UseCaseLabel(rec.UseCase)),
		Kind:          benchmark.JobKindBatch,
		Status:        benchmark.JobStatusPending,
		CreatedAt:     time.Now(),
		ModelIDs:      []string{rec.ModelID},
		BuildIDs:      []string{rec.BuildID},
		Presets:       presets,
		BaseProfile:   &profile,
		Sweeps:        in.Axes,
		AutotuneID:    rec.ID,
		AutotuneStage: stageKey,
	}
	for _, values := range in.Cells {
		for _, preset := range presets {
			cell := benchmark.JobCell{
				ModelID: rec.ModelID,
				BuildID: rec.BuildID,
				Preset:  preset,
				Status:  benchmark.CellStatusPending,
			}
			if len(values) > 0 {
				cell.SweepValues = copyValues(values)
			}
			job.Cells = append(job.Cells, cell)
		}
	}
	return job
}

// Estimating how long a stage takes. Both numbers are rough by nature —
// a first load reads the model from disk, a second reads it from the page
// cache — so they are deliberately on the slow side: a run that finishes
// early is a better surprise than one that does not.
const (
	// loadSecondsPerGiB is the model load a cell pays when the router
	// restarts for it.
	loadSecondsPerGiB = 4.0
	minLoadSeconds    = 20.0
	// presetSeconds is one preset's measurements: a few repetitions at
	// each prompt size.
	presetSeconds = 45.0
)

// EstimateMinutes is how long a set of cells takes for a model of
// sizeGiB, measuring presets per cell. secondsPerCell, when a previous
// stage measured one, replaces the estimate.
func EstimateMinutes(cells, presets int, sizeGiB, secondsPerCell float64) int {
	per := secondsPerCell
	if per <= 0 {
		load := sizeGiB * loadSecondsPerGiB
		if load < minLoadSeconds {
			load = minLoadSeconds
		}
		per = load + presetSeconds*float64(presets)
	}
	minutes := int((float64(cells)*per)/60 + 0.5)
	if minutes < 1 && cells > 0 {
		minutes = 1
	}
	return minutes
}
