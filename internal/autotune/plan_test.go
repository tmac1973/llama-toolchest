package autotune

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func baseInput() PlanInput {
	return PlanInput{
		Model: &models.Model{ID: "m", NLayers: 36, ContextLength: 131072},
		Base:  models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 32768, Threads: 8, FlashAttention: true},
		Cards: 1, Cores: 16,
		Finalists: map[string][]Candidate{},
	}
}

func values(cells []map[string]string, key string) map[string]bool {
	out := map[string]bool{}
	for _, c := range cells {
		out[c[key]] = true
	}
	return out
}

func TestPlanBatchStage(t *testing.T) {
	cells, axes, _ := PlanStage(StageBatch, baseInput())
	if len(cells) == 0 || len(cells[0]) != 0 {
		t.Fatalf("the first cell must be the starting profile: %v", cells)
	}
	if len(cells) != 11 {
		t.Errorf("cells = %d, want 11 (5 batch pairs × flash on/off, plus the baseline)", len(cells))
	}
	if got := values(cells, "split_mode"); len(got) != 1 {
		t.Errorf("split mode measured on one GPU: %v", got)
	}
	if got := values(cells, "threads"); len(got) != 1 {
		t.Errorf("threads measured with the whole model on the GPU: %v", got)
	}
	fields := map[string]bool{}
	for _, a := range axes {
		fields[a.Field] = true
	}
	if !fields["ubatch_size"] || !fields["flash_attention"] {
		t.Errorf("axes = %v", fields)
	}
}

// Flash attention off is not offered where llama.cpp would refuse to
// load it.
func TestPlanBatchKeepsFlashAttentionOnWhenItMust(t *testing.T) {
	in := baseInput()
	in.Base.KVCacheQuant = "q8_0"
	cells, _, _ := PlanStage(StageBatch, in)
	if got := values(cells, "flash_attention"); got["false"] {
		t.Error("flash attention off was measured with a quantized KV cache")
	}
}

func TestPlanBatchAddsSplitModeAndThreadsWhenTheyMatter(t *testing.T) {
	in := baseInput()
	in.Cards = 2
	cells, _, _ := PlanStage(StageBatch, in)
	if got := values(cells, "split_mode"); !got["layer"] || !got["tensor"] {
		t.Errorf("two GPUs should measure both split modes: %v", got)
	}

	in = baseInput()
	in.Base.CPUMoE = 12
	cells, _, _ = PlanStage(StageBatch, in)
	if got := values(cells, "threads"); len(got) < 3 {
		t.Errorf("expert layers on the CPU should measure thread counts: %v", got)
	}
}

// The speculative stage measures each method alone and every draft plus
// assist pair, and says what this machine cannot run.
func TestPlanSpecStage(t *testing.T) {
	in := baseInput()
	in.Model.NextNLayers = 1 // built-in MTP
	cells, _, skipped := PlanStage(StageSpec, in)

	specs := values(cells, "spec_type")
	if !specs["none"] {
		t.Error("the speculative stage must measure having it off")
	}
	assists := models.AssistModes()
	want := 1 + 1 + len(assists) + len(assists) // none + MTP + each assist + each pair
	if len(cells) != want {
		t.Errorf("cells = %d, want %d", len(cells), want)
	}
	pairs := 0
	for v := range specs {
		if strings.Contains(v, "draft-mtp+") {
			pairs++
		}
	}
	if pairs != len(assists) {
		t.Errorf("MTP was paired with %d assists, want %d", pairs, len(assists))
	}
	joined := strings.Join(skipped, " | ")
	for _, want := range []string{"Draft model", "EAGLE3", "DFlash", "DSpark"} {
		if !strings.Contains(joined, want) {
			t.Errorf("skipped does not mention %s: %v", want, skipped)
		}
	}
	if strings.Contains(joined, "MTP:") {
		t.Error("MTP was reported as unavailable although the model has draft layers")
	}
}

// An installed draft model is measured, named by its registry ID so the
// cell can resolve it.
func TestPlanSpecUsesInstalledDraftModels(t *testing.T) {
	in := baseInput()
	in.DraftCandidates = func(mode string) []models.DraftCandidate {
		if mode != "draft" {
			return nil
		}
		return []models.DraftCandidate{
			{ID: "big-draft", SizeGB: 2},
			{ID: "small-draft", SizeGB: 0.5},
			{ID: "third-draft", SizeGB: 3},
		}
	}
	cells, _, skipped := PlanStage(StageSpec, in)
	specs := values(cells, "spec_type")
	var named []string
	for v := range specs {
		if strings.Contains(v, "draft_model=") {
			named = append(named, v)
		}
	}
	if len(named) == 0 {
		t.Fatalf("no cell names a draft model: %v", specs)
	}
	joined := strings.Join(named, " ")
	if !strings.Contains(joined, "small-draft") || strings.Contains(joined, "third-draft") {
		t.Errorf("want the two smallest drafts only: %v", named)
	}
	if strings.Contains(strings.Join(skipped, " "), "Draft model:") {
		t.Error("a draft model is installed but was reported as missing")
	}
}

// The settings stage tunes what the previous stage chose, and leaves a
// finalist with no speculative decoding alone.
func TestPlanSpecParamsStage(t *testing.T) {
	in := baseInput()
	in.Finalists = map[string][]Candidate{StageSpec: {
		{Values: map[string]string{"spec_type": "draft-mtp+ngram-mod:assist_n_max=64,draft_max=6"}},
	}}
	cells, _, _ := PlanStage(StageSpecParams, in)
	if len(cells) == 0 || len(cells) > 12 {
		t.Fatalf("cells = %d, want between 1 and 12", len(cells))
	}
	draftMax := map[string]bool{}
	for v := range values(cells, "spec_type") {
		_, _, params := splitSpec(v)
		draftMax[params["draft_max"]] = true
	}
	if len(draftMax) < 3 {
		t.Errorf("draft lengths measured: %v, want several", draftMax)
	}

	in.Finalists = map[string][]Candidate{StageSpec: {{Values: map[string]string{"spec_type": "none"}}}}
	if cells, _, _ := PlanStage(StageSpecParams, in); len(cells) != 0 {
		t.Errorf("nothing to tune, but %d cells were planned", len(cells))
	}
}

// The confirming stage measures the finalists side by side against the
// starting profile, and stays small.
func TestPlanConfirmStage(t *testing.T) {
	in := baseInput()
	in.Finalists = map[string][]Candidate{
		StageBatch: {
			{Values: map[string]string{"ubatch_size": "1024"}, Scores: map[Goal]Score{GoalResponse: {Value: -5}}},
			{Values: map[string]string{"ubatch_size": "512"}, Scores: map[Goal]Score{GoalResponse: {Value: -6}}},
			{Values: map[string]string{"ubatch_size": "256"}, Scores: map[Goal]Score{GoalResponse: {Value: -9}}},
		},
		StageSpecParams: {
			{Values: map[string]string{"spec_type": "draft-mtp:draft_max=3"}},
			{Values: map[string]string{"spec_type": "draft-mtp+ngram-mod"}},
		},
	}
	cells, _, _ := PlanStage(StageConfirm, in)
	if len(cells) != 5 {
		t.Errorf("cells = %d, want 5 (2 batch × 2 speculative, plus the starting profile)", len(cells))
	}
	if len(cells[0]) != 0 {
		t.Error("the starting profile must be measured again alongside the finalists")
	}
	for _, c := range cells[1:] {
		if c["ubatch_size"] == "256" {
			t.Error("the slowest batch finalist should not reach the confirming stage")
		}
	}
}

func TestBuildStageJob(t *testing.T) {
	rec := &Autotune{ID: "at-1", ModelID: "m", BaseProfile: "Autoconfig", UseCase: UseMixed, BuildID: "b1"}
	cells, axes, _ := PlanStage(StageBatch, baseInput())
	job := BuildStageJob(rec, StageBatch, StageJobInput{
		Profile:   benchmark.BaseProfile{Name: "Autoconfig", Config: models.ModelConfig{ContextSize: 32768}},
		Cells:     cells,
		Axes:      axes,
		ModelName: "Qwen3.5-9B",
		JobID:     "job-1",
	})
	if job.AutotuneID != "at-1" || job.AutotuneStage != StageBatch {
		t.Errorf("job is not linked to its run: %+v", job)
	}
	if job.BaseProfile == nil || job.BaseProfile.Name != "Autoconfig" {
		t.Errorf("job does not measure the profile: %+v", job.BaseProfile)
	}
	if len(job.Cells) != len(cells)*2 {
		t.Errorf("cells = %d, want one per settings per preset (%d)", len(job.Cells), len(cells)*2)
	}
	if !strings.Contains(job.Name, "stage 1 of 4") || !strings.Contains(job.Name, "Qwen3.5-9B") {
		t.Errorf("job name = %q", job.Name)
	}
	if len(job.Cells[0].SweepValues) != 0 {
		t.Error("the first cell should carry no values: it is the starting profile")
	}
}

func TestEstimateMinutes(t *testing.T) {
	// 24 job cells of a 5 GiB model: each loads it and measures one
	// preset, so about 20 seconds of load plus 45 of measuring.
	if got := EstimateMinutes(24, 5, 0); got < 20 || got > 35 {
		t.Errorf("estimate = %d minutes, want roughly 26", got)
	}
	// A measured time per cell replaces the guess.
	if got := EstimateMinutes(10, 5, 30); got != 5 {
		t.Errorf("estimate with a measured 30s per cell = %d, want 5", got)
	}
	if got := EstimateMinutes(0, 5, 0); got != 0 {
		t.Errorf("no cells = %d minutes", got)
	}
}

// The quoted estimate must not be smaller than the run turns out to be:
// the middle stages build on several finalists, and a mixed workload
// measures every setting twice.
func TestEstimateRunIsNotOptimistic(t *testing.T) {
	in := baseInput()
	in.Model.NextNLayers = 1
	cells, minutes := EstimateRun(in, UseMixed, 5, 0)

	batch, _, _ := PlanStage(StageBatch, in)
	in.Finalists[StageBatch] = []Candidate{
		{Values: map[string]string{"ubatch_size": "1024"}},
		{Values: map[string]string{"ubatch_size": "512"}},
		{Values: map[string]string{"ubatch_size": "256"}},
	}
	spec, _, _ := PlanStage(StageSpec, in)
	if cells < len(batch)+len(spec) {
		t.Errorf("estimate of %d settings is below the %d the first two stages alone plan",
			cells, len(batch)+len(spec))
	}
	chatCells, chatMinutes := EstimateRun(in, UseChat, 5, 0)
	if chatCells != cells || chatMinutes >= minutes {
		t.Errorf("mixed (%d cells, %d min) should take longer than chat (%d, %d) for the same settings",
			cells, minutes, chatCells, chatMinutes)
	}
}

// llama.cpp refuses a tensor split without flash attention, so that pair
// must never be planned: it would load nothing and cost a model load.
func TestPlanBatchNeverPairsTensorSplitWithFlashAttentionOff(t *testing.T) {
	in := baseInput()
	in.Cards = 2
	cells, _, _ := PlanStage(StageBatch, in)
	for _, c := range cells {
		if c["split_mode"] == "tensor" && c["flash_attention"] == "false" {
			t.Fatalf("planned a combination llama.cpp refuses to load: %v", c)
		}
	}
	// Both are still measured, just not together.
	if got := values(cells, "split_mode"); !got["tensor"] {
		t.Error("the tensor split is no longer measured at all")
	}
	if got := values(cells, "flash_attention"); !got["false"] {
		t.Error("flash attention off is no longer measured at all")
	}
}

// A candidate with no response measurement must not displace a measured
// one from the confirming stage.
func TestPlanConfirmIgnoresUnmeasuredCandidates(t *testing.T) {
	in := baseInput()
	in.Finalists = map[string][]Candidate{
		StageBatch: {
			{Values: map[string]string{"ubatch_size": "4096"}}, // failed: no scores
			{Values: map[string]string{"ubatch_size": "1024"}, Scores: map[Goal]Score{GoalResponse: {Value: -5}}},
			{Values: map[string]string{"ubatch_size": "512"}, Scores: map[Goal]Score{GoalResponse: {Value: -6}}},
		},
	}
	cells, _, _ := PlanStage(StageConfirm, in)
	for _, c := range cells {
		if c["ubatch_size"] == "4096" {
			t.Fatalf("an unmeasured candidate reached the confirming stage: %v", cells)
		}
	}
}
