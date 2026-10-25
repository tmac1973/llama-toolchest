package autotune

import (
	"strings"
	"testing"
	"time"

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
	// 5 batch pairs with flash attention on, the smallest pair again
	// with it off, plus the baseline.
	if len(cells) != 7 {
		t.Errorf("cells = %d, want 7 (5 batch pairs, one flash-off probe, the baseline)", len(cells))
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

// Flash attention off is measured at the smallest prompt batch only.
// Without it the attention workspace grows with the prompt batch times
// the context size, so at a long context the larger prompt batches run
// out of memory and fail to load — half a stage spent on nothing.
func TestPlanBatchMeasuresFlashOffAtOnePromptBatchOnly(t *testing.T) {
	cells, _, _ := PlanStage(StageBatch, baseInput())
	var off []map[string]string
	for _, c := range cells {
		if c["flash_attention"] == "false" {
			off = append(off, c)
		}
	}
	if len(off) != 1 {
		t.Fatalf("flash attention off measured in %d cells, want 1: %v", len(off), off)
	}
	if off[0]["ubatch_size"] != "256" {
		t.Errorf("flash-off prompt batch = %q, want the smallest (256)", off[0]["ubatch_size"])
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

// Placement is measured in a stage of its own, on the batch finalists,
// and only where it can matter: the batch stage no longer multiplies its
// batch sizes by every split mode and thread count.
func TestPlacementStageMeasuresSplitAndThreadsWhenTheyMatter(t *testing.T) {
	finalists := []Candidate{
		{Values: map[string]string{"ubatch_size": "512"}},
		{Values: map[string]string{"ubatch_size": "1024"}},
	}

	in := baseInput()
	in.Cards = 2
	if batch, _, _ := PlanStage(StageBatch, in); len(values(batch, "split_mode")) > 1 || len(values(batch, "threads")) > 1 {
		t.Errorf("the batch stage still varies placement: %v", batch)
	}
	in.Finalists[StageBatch] = finalists
	cells, _, _ := PlanStage(StagePlacement, in)
	if got := values(cells, "split_mode"); !got["layer"] || !got["tensor"] {
		t.Errorf("two GPUs should measure both split modes: %v", got)
	}
	if got := values(cells, "ubatch_size"); !got["512"] || !got["1024"] {
		t.Errorf("placement is not measured on the batch finalists: %v", got)
	}

	in = baseInput()
	in.Base.CPUMoE = 12
	in.Finalists[StageBatch] = finalists
	cells, _, _ = PlanStage(StagePlacement, in)
	if got := values(cells, "threads"); len(got) < 3 {
		t.Errorf("expert layers on the CPU should measure thread counts: %v", got)
	}

	// MiniCPM on compute2: three GPUs and experts on the CPU. Batch and
	// placement were one stage of 31 settings; now 6 and then 2 x 6.
	in.Cards = 3
	batch, _, _ := PlanStage(StageBatch, baseInputWith(func(i *PlanInput) { i.Cards = 3; i.Base.CPUMoE = 4; i.Base.KVCacheQuant = "q8_0" }))
	cells, _, _ = PlanStage(StagePlacement, in)
	if len(batch) != 6 || len(cells) != 12 {
		t.Errorf("batch %d and placement %d settings, want 6 and 12", len(batch), len(cells))
	}

	// One GPU and nothing on the CPU: nothing to place.
	in = baseInput()
	in.Finalists[StageBatch] = finalists
	if cells, _, _ := PlanStage(StagePlacement, in); len(cells) != 0 {
		t.Errorf("placement planned %d settings with nothing to place", len(cells))
	}
}

func baseInputWith(f func(*PlanInput)) PlanInput {
	in := baseInput()
	f(&in)
	return in
}

// A quick run measures less at every stage that has a choice.
func TestQuickRunMeasuresLess(t *testing.T) {
	thorough := baseInputWith(func(i *PlanInput) { i.Model.NextNLayers = 1; i.Cards = 3; i.Base.CPUMoE = 4; i.MoECacheMiB = 4096 })
	quick := thorough
	quick.Quick = true
	quick.Finalists = map[string][]Candidate{}

	tb, _, _ := PlanStage(StageBatch, thorough)
	qb, _, _ := PlanStage(StageBatch, quick)
	if len(qb) >= len(tb) || values(qb, "flash_attention")["false"] {
		t.Errorf("quick batch stage = %d settings (flash off: %v), thorough %d", len(qb), values(qb, "flash_attention")["false"], len(tb))
	}
	quick.Finalists[StageSpec] = []Candidate{{Values: map[string]string{"spec_type": "draft-mtp:draft_max=6"}}}
	if cells, _, _ := PlanStage(StageSpecParams, quick); len(cells) != 0 {
		t.Errorf("a quick run fine-tuned the draft settings: %d settings", len(cells))
	}
	if got := len(moeCacheSizesFor(quick)); got != 2 {
		t.Errorf("quick expert cache sizes = %d, want 2", got)
	}
	tc, _ := EstimateRun(thorough, UseMixed, 27, 30)
	qc, _ := EstimateRun(quick, UseMixed, 27, 30)
	if qc*10 > tc*6 {
		t.Errorf("quick estimate %d settings against thorough %d: want well under two thirds", qc, tc)
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
	for _, want := range []string{"Draft Model", "EAGLE-3", "DFlash", "DSpark"} {
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

	if !values(cells, "spec_type")["draft-mtp+ngram-mod:assist_n_max=64,draft_max=6,draft_sampling=probabilistic"] {
		t.Errorf("probabilistic drafting not measured at the finalist's own settings: %v", values(cells, "spec_type"))
	}
	zero := 0.0
	in.Base.Temperature = &zero
	for v := range values(planStageCells(StageSpecParams, in), "spec_type") {
		if strings.Contains(v, "probabilistic") {
			t.Errorf("probabilistic drafting measured at temperature 0, where it drafts what greedy does: %s", v)
		}
	}
	in.Base.Temperature = nil

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
	if !strings.Contains(job.Name, "stage 1 of 6") || !strings.Contains(job.Name, "Qwen3.5-9B") {
		t.Errorf("job name = %q", job.Name)
	}
	if len(job.Cells[0].SweepValues) != 0 {
		t.Error("the first cell should carry no values: it is the starting profile")
	}
}

func TestEstimateMinutes(t *testing.T) {
	// 24 job cells of a 5 GiB model: each reloads it, mostly from the page
	// cache, and measures one preset — about 5 seconds and 20.
	if got := EstimateMinutes(24, 5, 0); got < 8 || got > 12 {
		t.Errorf("estimate = %d minutes, want roughly 10", got)
	}
	// compute2's 20 GiB MoE took 23 seconds a cell; the guess stays on
	// the slow side of that without being five times it.
	if per := defaultCellSeconds(20); per < 25 || per > 60 {
		t.Errorf("a 20 GiB model is guessed at %.0f s a cell, against 23 measured", per)
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
	in.Finalists[StageBatch] = []Candidate{
		{Values: map[string]string{"ubatch_size": "256", "flash_attention": "false"}},
		{Values: map[string]string{"ubatch_size": "512"}},
	}
	placed, _, _ := PlanStage(StagePlacement, in)
	cells = append(cells, placed...)
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

// The registry lists every installed model for the head-based methods,
// because its picker cannot tell a converted head from anything else and
// the user chooses. Autotune chooses on its own, so it must not offer an
// ordinary chat model as an EAGLE3 head: llama.cpp refuses to load it,
// and the run spends minutes failing once per method.
func TestPlanSpecSkipsHeadMethodsWithoutAHead(t *testing.T) {
	in := baseInput()
	in.Model.NextNLayers = 1
	in.DraftCandidates = func(mode string) []models.DraftCandidate {
		if mode == "draft" {
			return nil
		}
		// What the registry really returns for a head-based mode: every
		// installed model, filters deliberately not applied.
		return []models.DraftCandidate{
			{ID: "unsloth--Qwen3.5-4B-GGUF--Qwen3.5-4B-Q4_K_M", Filename: "Qwen3.5-4B-Q4_K_M.gguf", SizeGB: 2.5},
		}
	}
	cells, _, skipped := PlanStage(StageSpec, in)

	for v := range values(cells, "spec_type") {
		for _, mode := range []string{"draft-eagle3", "draft-dflash", "draft-dspark"} {
			if strings.Contains(v, mode) {
				t.Errorf("%s was measured with a chat model as its head: %q", mode, v)
			}
		}
	}
	joined := strings.Join(skipped, " | ")
	for _, name := range []string{"EAGLE-3", "DFlash", "DSpark"} {
		if !strings.Contains(joined, name) {
			t.Errorf("skipped does not mention %s: %v", name, skipped)
		}
	}
}

// A head made for another model is not measured: the MiniCPM-V-4.7 DSpark
// head has the shape of Qwen3.6-35B-A3B, which MiniCPM-V is built on, and
// every cell that drafted Qwen3.6 with it failed to load. One the profile
// already uses is measured all the same.
func TestPlanSpecSkipsAnotherModelsHead(t *testing.T) {
	in := baseInput()
	in.Model.ModelID = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
	in.Model.Filename = "Qwen3.6-35B-A3B-UD-IQ4_NL.gguf"
	other := models.DraftCandidate{
		ID:       "openbmb--MiniCPM-V-4.7-35B-A3B-DSpark-gguf--dspark-MiniCPM-V-4_7-35B-A3B-Q8_0",
		Filename: "dspark-MiniCPM-V-4_7-35B-A3B-Q8_0.gguf", FilePath: "/data/models/dspark.gguf", SizeGB: 0.4,
	}
	in.DraftCandidates = func(mode string) []models.DraftCandidate {
		if mode == "draft-dspark" {
			return []models.DraftCandidate{other}
		}
		return nil
	}
	cells, _, skipped := PlanStage(StageSpec, in)
	for v := range values(cells, "spec_type") {
		if strings.Contains(v, "draft-dspark") {
			t.Errorf("another model's head was measured: %q", v)
		}
	}
	if !strings.Contains(strings.Join(skipped, " | "), "made for other models") {
		t.Errorf("skipped does not say why DSpark was left out: %v", skipped)
	}

	in.Base.DraftModelPath = other.FilePath
	cells, _, _ = PlanStage(StageSpec, in)
	found := false
	for v := range values(cells, "spec_type") {
		found = found || strings.Contains(v, "draft-dspark")
	}
	if !found {
		t.Error("the head the profile uses was not measured")
	}
}

// A file whose name says which method it belongs to is measured.
func TestPlanSpecUsesAnInstalledHead(t *testing.T) {
	in := baseInput()
	in.Model.ModelID = "Qwen/Qwen3.5-9B-GGUF"
	in.Model.Filename = "Qwen3.5-9B-Q4_K_M.gguf"
	in.DraftCandidates = func(mode string) []models.DraftCandidate {
		if mode == "draft" {
			return nil
		}
		return []models.DraftCandidate{
			{ID: "chat-model", Filename: "Qwen3.5-4B-Q4_K_M.gguf", SizeGB: 2.5},
			{ID: "head", Filename: "Qwen3.5-9B-EAGLE3-Q8_0.gguf", SizeGB: 0.3},
		}
	}
	cells, _, _ := PlanStage(StageSpec, in)

	var found string
	for v := range values(cells, "spec_type") {
		if strings.HasPrefix(v, "draft-eagle3") {
			found = v
		}
	}
	if found == "" {
		t.Fatal("an installed EAGLE3 head was not measured")
	}
	if !strings.Contains(found, "head") || strings.Contains(found, "chat-model") {
		t.Errorf("EAGLE3 used %q, want the head rather than the chat model", found)
	}
}

// A drafter loaded from its own file is measured with the model's own
// cache type when the profile stores its cache at 8 bits. At full
// precision the drafter's cache, sized for the model's whole context,
// did not fit on the machine this is from, so every such cell failed
// instead of being measured.
func TestPlanSpecMeasuresADraftModelWithTheModelsCacheType(t *testing.T) {
	withDrafts := func(in PlanInput) PlanInput {
		in.Model.NextNLayers = 1 // MTP too, to check it is left alone
		in.DraftCandidates = func(mode string) []models.DraftCandidate {
			if mode != "draft" {
				return nil
			}
			return []models.DraftCandidate{{ID: "small-draft", SizeGB: 2.5}}
		}
		return in
	}

	quantized := withDrafts(baseInput())
	quantized.Base.KVCacheQuant = "q8_0"
	cells, _, _ := PlanStage(StageSpec, quantized)
	var sawDraft bool
	for v := range values(cells, "spec_type") {
		hasCache := strings.Contains(v, "draft_cache=q8_0")
		switch {
		case strings.Contains(v, "draft_model=small-draft"):
			sawDraft = true
			if !hasCache {
				t.Errorf("draft model measured with a full-precision cache: %q", v)
			}
		case hasCache:
			t.Errorf("a value that loads no drafter file carries a draft cache: %q", v)
		}
	}
	if !sawDraft {
		t.Fatal("no cell measures the draft model")
	}
	if got := describeSpec("draft:draft_cache=q8_0,draft_max=16,draft_model=org--small-draft"); !strings.Contains(got, "q8_0 draft cache") {
		t.Errorf("label %q does not mention the draft cache", got)
	}

	// A profile with a full-precision cache has room to spare; the
	// drafter's is left at llama.cpp's default.
	full := withDrafts(baseInput())
	cells, _, _ = PlanStage(StageSpec, full)
	for v := range values(cells, "spec_type") {
		if strings.Contains(v, "draft_cache=") {
			t.Errorf("draft cache set although the profile keeps its own cache at full precision: %q", v)
		}
	}

	// A profile that chose a draft cache type keeps its choice.
	chosen := withDrafts(baseInput())
	chosen.Base.KVCacheQuant, chosen.Base.DraftKVCacheQuant = "q8_0", "q4_0"
	cells, _, _ = PlanStage(StageSpec, chosen)
	for v := range values(cells, "spec_type") {
		if strings.Contains(v, "draft_cache=") {
			t.Errorf("the profile's own draft cache setting was overridden: %q", v)
		}
	}
}

// The expert-cache stage measures each batch finalist with the cache off
// and at a quarter, a half and all of the room left, and the speculative
// stage builds on its finalists.
func TestPlanMoECacheStage(t *testing.T) {
	in := baseInput()
	if cells, _, _ := PlanStage(StageMoECache, in); len(cells) != 0 {
		t.Errorf("no room for a cache: %v, want nothing to measure", cells)
	}

	in.MoECacheMiB = 4096
	in.Finalists[StageBatch] = []Candidate{
		{Values: map[string]string{"ubatch_size": "512"}},
		{Values: map[string]string{"ubatch_size": "1024"}},
	}
	cells, _, _ := PlanStage(StageMoECache, in)
	if len(cells) != 8 {
		t.Fatalf("cells = %d, want 2 finalists x 4 sizes: %v", len(cells), cells)
	}
	if got := values(cells, "moe_cache_mib"); len(got) != 4 || !got["0"] || !got["1024"] || !got["2048"] || !got["4096"] {
		t.Errorf("sizes = %v, want 0, 1024, 2048 and 4096", got)
	}
	if got := values(cells, "ubatch_size"); !got["512"] || !got["1024"] {
		t.Errorf("the batch finalists are not carried: %v", got)
	}

	in.Finalists[StageMoECache] = []Candidate{{Values: map[string]string{"ubatch_size": "512", "moe_cache_mib": "2048"}}}
	spec, _, _ := PlanStage(StageSpec, in)
	if got := values(spec, "moe_cache_mib"); len(got) != 1 || !got["2048"] {
		t.Errorf("speculative stage cache sizes = %v, want the cache stage's finalist", got)
	}
}

func TestMoECacheSizes(t *testing.T) {
	for _, tc := range []struct {
		room int
		want []int
	}{
		{0, nil},
		{512, []int{256, 512}},
		{768, []int{256, 768}},
		{10000, []int{2304, 4864, 9984}},
	} {
		got := moeCacheSizes(tc.room)
		if len(got) != len(tc.want) {
			t.Errorf("moeCacheSizes(%d) = %v, want %v", tc.room, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("moeCacheSizes(%d) = %v, want %v", tc.room, got, tc.want)
				break
			}
		}
	}
}

// planStageCells is PlanStage's cells alone.
func planStageCells(key string, in PlanInput) []map[string]string {
	cells, _, _ := PlanStage(key, in)
	return cells
}

// The time a cell takes comes from this machine's history when it has
// one: the model's own latest measurement, or the default scaled by how
// far it was off for the models that were measured.
func TestCellSecondsFromHistory(t *testing.T) {
	now := time.Now()
	sizes := map[string]float64{"lfm": 20, "dolphin": 18}
	sizeOf := func(id string) float64 { return sizes[id] }
	recs := []*Autotune{
		{ModelID: "lfm", SecondsPerCell: 24, CreatedAt: now.Add(-2 * time.Hour)},
		{ModelID: "minicpm", SecondsPerCell: 30, CreatedAt: now.Add(-3 * time.Hour)},
		{ModelID: "minicpm", SecondsPerCell: 33, CreatedAt: now.Add(-1 * time.Hour)},
		{ModelID: "minicpm", CreatedAt: now},                   // cancelled before measuring
		{ModelID: "gone", SecondsPerCell: 500, CreatedAt: now}, // no longer installed
	}
	if got := CellSecondsFromHistory(recs, "minicpm", 27, sizeOf); got != 33 {
		t.Errorf("own history = %.1f s, want the latest measurement, 33", got)
	}
	// A model never run here: the default for its size, scaled by how
	// far the default was off for LFM (24 measured against 40 guessed).
	want := 24 / defaultCellSeconds(20) * defaultCellSeconds(27)
	if got := CellSecondsFromHistory(recs, "new", 27, sizeOf); got < want-0.01 || got > want+0.01 {
		t.Errorf("from other models = %.1f s, want %.1f", got, want)
	}
	if got := CellSecondsFromHistory(nil, "new", 27, sizeOf); got != 0 {
		t.Errorf("no history = %.1f s, want 0 so the default stands", got)
	}
}
