package autoconfig

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// threeCards is the machine the test load was written for: three 16 GiB
// cards, where a plan can fit in total and still overflow one of them.
func threeCards() models.Hardware {
	return models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, VRAMTotalMiB: 16376}, {Index: 1, VRAMTotalMiB: 16376}, {Index: 2, VRAMTotalMiB: 16376},
	}, LogicalCores: 16, RAMTotalMiB: 64 * 1024}
}

// verifier is a machine for the test load to run on: runs decides whether
// a config loads, and every config it is asked about is recorded.
type verifier struct {
	runs  func(models.ModelConfig) Check
	err   error
	tried []models.ModelConfig
}

func (v *verifier) verify(_ context.Context, _ string, cfg models.ModelConfig) (Check, error) {
	v.tried = append(v.tried, cfg)
	if v.err != nil {
		return Check{}, v.err
	}
	return v.runs(cfg), nil
}

func outOfMemoryOn(device int) Check {
	return Check{OutOfMemory: true, Device: device, Reason: "CUDA error: out of memory"}
}

func runWith(t *testing.T, hw models.Hardware, v *verifier) *Result {
	t.Helper()
	res, err := Run(context.Background(), Deps{Registry: runRegistry(t), Hardware: hw, Verify: v.verify},
		runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	return res
}

func noteFor(res *Result, field string) string {
	var out []string
	for _, n := range res.Notes {
		if n.Field == field {
			out = append(out, n.Origin+": "+n.Reason)
		}
	}
	return strings.Join(out, " | ")
}

// Settings that load and answer are proposed as they are, and the review
// can say they were checked.
func TestVerifyPassesAProposalThatRuns(t *testing.T) {
	v := &verifier{runs: func(models.ModelConfig) Check { return Check{OK: true} }}
	res := runWith(t, threeCards(), v)

	if res.Check.Status != VerifyPassed || res.Check.Attempts != 1 || res.Check.Adjusted != 0 {
		t.Errorf("check = %+v, want passed on the first load with nothing adjusted", res.Check)
	}
	if len(v.tried) != 1 {
		t.Fatalf("loaded %d times, want once", len(v.tried))
	}
	if res.Proposed.GPUAssign == "custom" || res.Proposed.UBatchSize != 0 {
		t.Errorf("a proposal that ran was changed: %+v", res.Proposed)
	}
	// What was loaded is what is proposed, draft layers included.
	if v.tried[0].SpecType != "draft-mtp" {
		t.Errorf("the test load ran with spec_type %q; it must load the finished proposal", v.tried[0].SpecType)
	}
}

// The case this exists for: the plan fits in total, and one card runs
// out while the others have room. Moving layers off that card costs
// nothing, so it is what is tried first.
func TestVerifyMovesLayersOffTheCardThatRanOut(t *testing.T) {
	v := &verifier{runs: func(cfg models.ModelConfig) Check {
		if cfg.TensorSplit == "" {
			return outOfMemoryOn(2)
		}
		return Check{OK: true}
	}}
	res := runWith(t, threeCards(), v)

	if res.Check.Status != VerifyPassed || res.Check.Attempts != 2 || res.Check.Adjusted != 1 {
		t.Fatalf("check = %+v, want passed on the second load with one setting adjusted", res.Check)
	}
	p := res.Proposed
	if p.GPUAssign != "custom" || p.TensorSplit != "20,20,17" || p.SplitMode != "layer" {
		t.Errorf("placement = %q / %q / %q, want a custom layer split of 20,20,17", p.GPUAssign, p.TensorSplit, p.SplitMode)
	}
	if p.UBatchSize != 0 || p.ContextSize != 8192 {
		t.Errorf("the prompt batch or the context was reduced (%d / %d) when moving layers was enough", p.UBatchSize, p.ContextSize)
	}
	note := noteFor(res, "gpu_assign")
	for _, want := range []string{"test load", "GPU 2", "20,20,17"} {
		if !strings.Contains(note, want) {
			t.Errorf("gpu_assign note %q does not mention %q", note, want)
		}
	}
	if res.Fit.Config.TensorSplit != p.TensorSplit {
		t.Error("the fit result still describes the plan that did not run")
	}
}

// A load that answered but left a card nearly full is adjusted like one
// that ran out, and the note says what actually happened.
func TestVerifyAdjustsALoadThatLeftACardNearlyFull(t *testing.T) {
	v := &verifier{runs: func(cfg models.ModelConfig) Check {
		if cfg.TensorSplit == "" {
			return Check{OutOfMemory: true, LowMemory: true, Device: 0, Reason: "it left only 497 MiB free on GPU 0"}
		}
		return Check{OK: true}
	}}
	res := runWith(t, threeCards(), v)
	if res.Check.Status != VerifyPassed || res.Check.Adjusted != 1 {
		t.Fatalf("check = %+v, want passed after one adjustment", res.Check)
	}
	note := noteFor(res, "gpu_assign")
	if !strings.Contains(note, "left too little GPU memory free") || strings.Contains(note, "ran out of memory") {
		t.Errorf("note = %q, want it to say the load left too little memory free", note)
	}
}

// When nothing fits, the steps are taken in order of what they cost:
// layers moved (nothing), a smaller prompt batch (some speed), then the
// context (what the model can hold). And the run ends, with the proposal
// left as estimated and marked as not working.
func TestVerifyStepsDownInOrderOfCostAndGivesUp(t *testing.T) {
	v := &verifier{runs: func(models.ModelConfig) Check { return outOfMemoryOn(2) }}
	res := runWith(t, threeCards(), v)

	// Five loads: at 4,096 tokens there is nothing smaller left to try.
	if res.Check.Status != VerifyFailed || res.Check.Attempts != 5 {
		t.Fatalf("check = %+v, want failed after 5 loads", res.Check)
	}
	if !strings.Contains(res.Check.Reason, "out of memory") {
		t.Errorf("reason = %q, want what the server reported", res.Check.Reason)
	}
	if len(v.tried) != 5 {
		t.Fatalf("loaded %d times, want 5", len(v.tried))
	}
	type step struct {
		split  string
		ubatch int
		ctx    int
	}
	var got []step
	for _, c := range v.tried {
		got = append(got, step{c.TensorSplit, c.UBatchSize, c.ContextSize})
	}
	want := []step{
		{"", 0, 8192},           // the plan
		{"20,20,17", 0, 8192},   // layers off the card that ran out
		{"20,20,14", 0, 8192},   // and again
		{"20,20,14", 256, 8192}, // then a smaller prompt batch
		{"20,20,14", 256, 4096}, // then half the context
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("load %d = %+v, want %+v", i+1, got[i], w)
		}
	}
	// Nothing ran, so nothing the check tried is proposed.
	if res.Proposed.GPUAssign == "custom" || res.Proposed.UBatchSize != 0 || res.Proposed.ContextSize != 8192 {
		t.Errorf("a plan that never ran was proposed: %+v", res.Proposed)
	}
	if noteFor(res, "gpu_assign") != "" && strings.Contains(noteFor(res, "gpu_assign"), "test load") {
		t.Error("a note explains an adjustment that was not kept")
	}
}

// A failed load takes minutes, so the number of them is bounded however
// much context there is left to halve.
func TestVerifyStopsAtTheLimitOnLoads(t *testing.T) {
	v := &verifier{runs: func(models.ModelConfig) Check { return outOfMemoryOn(2) }}
	res, err := Run(context.Background(), Deps{Registry: runRegistry(t), Hardware: threeCards(), Verify: v.verify},
		runModelID, models.ContextMax)
	if err != nil {
		t.Fatal(err)
	}
	if res.Check.Status != VerifyFailed || len(v.tried) != maxChecks {
		t.Errorf("check = %+v after %d loads, want failed after %d", res.Check, len(v.tried), maxChecks)
	}
}

// A reduced context is explained once, by the note that says why, and
// the fit's note for the size it no longer has is dropped.
func TestVerifyReplacesTheFitsNoteForASizeItReduced(t *testing.T) {
	v := &verifier{runs: func(cfg models.ModelConfig) Check {
		if cfg.ContextSize > 4096 {
			return outOfMemoryOn(0)
		}
		return Check{OK: true}
	}}
	// One card: there is nowhere to move layers to.
	res := runWith(t, gpu24(), v)

	if res.Check.Status != VerifyPassed {
		t.Fatalf("check = %+v", res.Check)
	}
	if res.Proposed.ContextSize != 4096 || res.Proposed.UBatchSize != 256 {
		t.Errorf("context / prompt batch = %d / %d, want 4096 / 256", res.Proposed.ContextSize, res.Proposed.UBatchSize)
	}
	if res.Proposed.GPUAssign == "custom" {
		t.Error("layers were moved on a machine with one card")
	}
	note := noteFor(res, "context_size")
	if !strings.Contains(note, "test load") || !strings.Contains(note, "4,096") {
		t.Errorf("context note = %q, want the test load's explanation", note)
	}
	if strings.Contains(note, "hardware fit") {
		t.Errorf("context note = %q: the fit's note for the larger size is still there", note)
	}
	if res.Check.Adjusted != 2 {
		t.Errorf("adjusted = %d, want 2 (prompt batch and context)", res.Check.Adjusted)
	}
}

// A failure that is not about memory is not one a smaller plan fixes.
// One load, and the review says what went wrong.
func TestVerifyDoesNotStepDownForOtherFailures(t *testing.T) {
	v := &verifier{runs: func(models.ModelConfig) Check {
		return Check{Reason: "error loading model: unknown architecture", Device: -1}
	}}
	res := runWith(t, threeCards(), v)

	if res.Check.Status != VerifyFailed || res.Check.Attempts != 1 {
		t.Errorf("check = %+v, want failed after one load", res.Check)
	}
	if !strings.Contains(res.Check.Reason, "unknown architecture") {
		t.Errorf("reason = %q", res.Check.Reason)
	}
	if len(v.tried) != 1 {
		t.Errorf("loaded %d times for a failure a smaller plan cannot fix", len(v.tried))
	}
}

// When the check cannot be run at all, the proposal is still returned,
// marked as not checked, with the reason.
func TestVerifyThatCannotRunLeavesTheProposalUnchecked(t *testing.T) {
	v := &verifier{err: errors.New("a benchmark started and is using the GPU")}
	res := runWith(t, threeCards(), v)

	if res.Check.Status != VerifySkipped || !strings.Contains(res.Check.Reason, "benchmark") {
		t.Errorf("check = %+v, want skipped with the reason", res.Check)
	}
	if res.Proposed.ContextSize != 8192 {
		t.Errorf("proposal lost: %+v", res.Proposed)
	}
}

// Without a verifier nothing is loaded and nothing is claimed.
func TestRunWithoutAVerifierChecksNothing(t *testing.T) {
	res, err := Run(context.Background(), Deps{Registry: runRegistry(t), Hardware: threeCards()}, runModelID, models.ContextShort)
	if err != nil {
		t.Fatal(err)
	}
	if res.Check.Status != "" {
		t.Errorf("check status = %q with no verifier", res.Check.Status)
	}
}

func TestMoveLayersOff(t *testing.T) {
	layer := models.ModelConfig{GPUAssign: "all", SplitMode: "layer"}
	tests := []struct {
		name    string
		cfg     models.ModelConfig
		device  int
		gpus    int
		want    string // the new split; "" when nothing is moved
		wantDev int
	}{
		{"the named card", layer, 1, 3, "20,17,20", 1},
		{"a second move from the same card", models.ModelConfig{GPUAssign: "custom", SplitMode: "layer", TensorSplit: "20,20,17"}, 2, 3, "20,20,14", 2},
		{"a second move from another card", models.ModelConfig{GPUAssign: "custom", SplitMode: "layer", TensorSplit: "20,20,17"}, 1, 3, "20,17,17", 1},
		{"no card named, model drafts: the last card", models.ModelConfig{GPUAssign: "all", SplitMode: "layer", SpecType: "draft-mtp"}, -1, 3, "20,20,17", 2},
		{"no card named, no draft context", layer, -1, 3, "", 0},
		{"one card", layer, 0, 1, "", 0},
		{"tensor parallelism", models.ModelConfig{GPUAssign: "tensor-3", SplitMode: "tensor"}, 2, 3, "", 0},
		{"a card left out of the assignment", models.ModelConfig{GPUAssign: "0-1", SplitMode: "layer", TensorSplit: "1,1,0"}, 1, 3, "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			next, dev, ok := moveLayersOff(tt.cfg, tt.device, tt.gpus)
			if tt.want == "" {
				if ok {
					t.Errorf("moved layers (split %q) where there was nothing to move", next.TensorSplit)
				}
				return
			}
			if !ok || next.TensorSplit != tt.want || dev != tt.wantDev {
				t.Errorf("split = %q from GPU %d (ok=%v), want %q from GPU %d", next.TensorSplit, dev, ok, tt.want, tt.wantDev)
			}
			if next.GPUAssign != "custom" {
				t.Errorf("gpu_assign = %q; only \"custom\" keeps a written split when the config is saved", next.GPUAssign)
			}
		})
	}
}

// Draft layers are turned on after the fit is planned, and the draft
// context they need is memory too. A fit planned without it chooses a
// context that only fits while drafting is off.
func TestFitIsPlannedWithTheDraftContext(t *testing.T) {
	estimate := func(mtpLayers int) (float64, int) {
		res, err := Run(context.Background(), Deps{Registry: runRegistryWithMTP(t, mtpLayers), Hardware: gpu24()},
			runModelID, models.ContextMax)
		if err != nil {
			t.Fatal(err)
		}
		return res.Fit.EstimateGiB, res.Proposed.ContextSize
	}
	with, ctxWith := estimate(1)
	without, ctxWithout := estimate(0)
	if ctxWith != ctxWithout {
		t.Skipf("the two plans chose different contexts (%d and %d), so their estimates are not comparable", ctxWith, ctxWithout)
	}
	if with <= without {
		t.Errorf("estimate with draft layers = %.2f GiB, without = %.2f GiB; the draft context was not counted", with, without)
	}
}
