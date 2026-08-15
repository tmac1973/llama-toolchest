package api

import (
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

func baseConfig() models.ModelConfig {
	return models.ModelConfig{
		Enabled:        true,
		ContextSize:    8192,
		GPULayers:      999,
		Threads:        8,
		FlashAttention: true,
		KVCacheQuant:   "q8_0",
		GPUAssign:      "all",
		Aliases:        []string{"keep-me"},
	}
}

func TestApplySnapshotOverridesSetFields(t *testing.T) {
	snap := benchmark.ConfigSnapshot{
		ContextSize:    65536,
		GPULayers:      50,
		Threads:        16,
		GPUAssign:      "all",
		TensorSplit:    "1,1",
		KVCacheQuant:   "q4_0",
		FlashAttention: true,
	}
	got := applySnapshotToConfig(baseConfig(), snap)

	if got.ContextSize != 65536 {
		t.Errorf("ContextSize = %d, want 65536", got.ContextSize)
	}
	if got.GPULayers != 50 {
		t.Errorf("GPULayers = %d, want 50", got.GPULayers)
	}
	if got.Threads != 16 {
		t.Errorf("Threads = %d, want 16", got.Threads)
	}
	if got.TensorSplit != "1,1" {
		t.Errorf("TensorSplit = %q, want 1,1", got.TensorSplit)
	}
	if got.KVCacheQuant != "q4_0" {
		t.Errorf("KVCacheQuant = %q, want q4_0", got.KVCacheQuant)
	}
}

// Fields the snapshot doesn't model must survive untouched — the
// override applies to a copy of the saved config, not a fresh struct.
func TestApplySnapshotPreservesUnmodeledFields(t *testing.T) {
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{
		ContextSize: 4096, GPUAssign: "all",
	})

	if !got.Enabled {
		t.Error("Enabled was dropped")
	}
	if len(got.Aliases) != 1 || got.Aliases[0] != "keep-me" {
		t.Errorf("Aliases = %v, want [keep-me]", got.Aliases)
	}
	if got.GPUAssign != "all" {
		t.Errorf("GPUAssign = %q, want all", got.GPUAssign)
	}
}

// Zero is a value, not "unset". ResolveModel seeds the snapshot from the
// model's saved config and applyOverrides replaces only what the job set,
// so the snapshot is authoritative for every field it models.
//
// This test previously asserted the opposite, which locked in a bug:
// sweeping gpu_layers to 0 (the curated "CPU only" choice) was silently
// discarded while the run still recorded 0 as applied — the same
// mislabeled-result failure this whole mechanism exists to prevent.
func TestApplySnapshotAppliesZeroValues(t *testing.T) {
	got := applySnapshotToConfig(baseConfig(), benchmark.ConfigSnapshot{
		GPULayers:   0,
		ContextSize: 4096,
		Threads:     8,
	})

	if got.GPULayers != 0 {
		t.Errorf("GPULayers = %d, want the overridden 0 (CPU only)", got.GPULayers)
	}
	if got.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", got.ContextSize)
	}
}

// A snapshot carrying the saved values reproduces them exactly, which is
// the no-override case.
func TestApplySnapshotRoundTripsSavedValues(t *testing.T) {
	base := baseConfig()
	got := applySnapshotToConfig(base, benchmark.ConfigSnapshot{
		GPULayers:      base.GPULayers,
		ContextSize:    base.ContextSize,
		Threads:        base.Threads,
		KVCacheQuant:   base.KVCacheQuant,
		GPUAssign:      base.GPUAssign,
		FlashAttention: base.FlashAttention,
	})

	if got.ContextSize != 8192 || got.GPULayers != 999 || got.Threads != 8 {
		t.Errorf("round trip changed values: %+v", got)
	}
	if got.KVCacheQuant != "q8_0" {
		t.Errorf("KVCacheQuant = %q, want q8_0", got.KVCacheQuant)
	}
}

// The input must not be mutated — callers pass a value copy of the
// registry's live config and the registry must not observe the change.
func TestApplySnapshotDoesNotMutateInput(t *testing.T) {
	base := baseConfig()
	_ = applySnapshotToConfig(base, benchmark.ConfigSnapshot{ContextSize: 65536, GPULayers: 1})

	if base.ContextSize != 8192 || base.GPULayers != 999 {
		t.Errorf("input mutated: ctx=%d ngl=%d", base.ContextSize, base.GPULayers)
	}
}

// A benchmark job's build and config travel as explicit start options,
// so they can only reach the router through a start the job itself
// makes. Ambient state was the root of a family of bugs: an interactive
// restart picked up the job's config, the job's build was persisted to
// disk, and the launch snapshot needed a flag to suppress it.
func TestJobRouterOptionsAreExplicit(t *testing.T) {
	e := &jobEnv{}

	if e.routerOwnedByJob() {
		t.Error("no job has taken the router")
	}
	if opt := e.jobRouterOptions(); opt.buildID != "" || opt.overrides != nil {
		t.Errorf("idle options should be zero, got %+v", opt)
	}

	e.jobBuildID = "build-B"
	e.jobConfigs = map[string]*models.ModelConfig{"m": {ContextSize: 4096}}
	e.ownsRouter = true

	opt := e.jobRouterOptions()
	if opt.buildID != "build-B" {
		t.Errorf("buildID = %q, want build-B", opt.buildID)
	}
	if opt.overrides == nil || opt.overrides["m"].ContextSize != 4096 {
		t.Errorf("overrides = %+v, want the job's substitute config", opt.overrides)
	}
	if !e.routerOwnedByJob() {
		t.Error("the job holds the router")
	}
}

// The zero options are what every interactive start uses, and they must
// mean "the user's build and the user's preset".
func TestZeroRouterOptionsMeanUserSettings(t *testing.T) {
	var opt routerOptions
	if opt.buildID != "" {
		t.Error("empty buildID must fall back to the saved selection")
	}
	if opt.overrides != nil {
		t.Error("nil overrides must use the user's preset")
	}
}

// Releasing must clear ownership without disturbing anything else, and
// clearing intent must not by itself release the router — the restore
// restart has to succeed first.
func TestReleaseRouterClearsOwnership(t *testing.T) {
	e := &jobEnv{ownsRouter: true, jobBuildID: "b", jobConfigs: map[string]*models.ModelConfig{"m": {}}}
	e.releaseRouter()
	if e.routerOwnedByJob() {
		t.Error("ownership should be cleared")
	}
}

// gpu_assign only becomes real llama-server flags through
// models.ResolveGPUAssign — writeConfigParams emits tensor-split,
// split-mode and main-gpu, and never reads GPUAssign. Without resolving
// it, a gpu_assign sweep generated byte-identical presets for every
// point and reported one measurement under several labels.
func TestGPUAssignSweepResolvesToRealFlags(t *testing.T) {
	base := models.ModelConfig{GPUAssign: "all", SplitMode: "layer", MainGPU: 0}

	out := applySnapshotToConfig(base, benchmark.ConfigSnapshot{GPUAssign: "0"})
	resolveGPUAssignment(&out, base, 2)

	if out.GPUAssign != "0" {
		t.Fatalf("GPUAssign = %q, want the swept 0", out.GPUAssign)
	}
	wantTS, wantSM, wantMG := models.ResolveGPUAssign("0", 2)
	if out.TensorSplit != wantTS || out.SplitMode != wantSM || out.MainGPU != wantMG {
		t.Errorf("resolved to (%q, %q, %d), want (%q, %q, %d)",
			out.TensorSplit, out.SplitMode, out.MainGPU, wantTS, wantSM, wantMG)
	}

	// The whole point: the two sweep points must not produce the same
	// preset input.
	other := applySnapshotToConfig(base, benchmark.ConfigSnapshot{GPUAssign: "all"})
	resolveGPUAssignment(&other, base, 2)
	if other.TensorSplit == out.TensorSplit && other.SplitMode == out.SplitMode && other.MainGPU == out.MainGPU {
		t.Error("gpu_assign sweep points resolved identically; the sweep would be a no-op")
	}
}

// A job that doesn't touch gpu_assign must leave the model's saved
// split-mode and main-gpu alone.
func TestGPUAssignUnchangedLeavesDerivedFields(t *testing.T) {
	base := models.ModelConfig{GPUAssign: "custom", TensorSplit: "3,1", SplitMode: "tensor", MainGPU: 1}
	out := applySnapshotToConfig(base, benchmark.ConfigSnapshot{GPUAssign: "custom", TensorSplit: "3,1"})
	resolveGPUAssignment(&out, base, 2)

	if out.TensorSplit != "3,1" || out.SplitMode != "tensor" || out.MainGPU != 1 {
		t.Errorf("untouched assignment was rewritten: %q %q %d", out.TensorSplit, out.SplitMode, out.MainGPU)
	}
}

// gpu_assign and tensor_split are two ways to say the same thing, and
// only tensor_split reaches llama-server. Setting both is contradictory:
// honouring either one silently made some cells identical while
// reporting different values. The combination is refused when the job is
// defined, so resolution never has to guess.
func TestContradictoryGPUPlacementIsRefused(t *testing.T) {
	assign, split := "0-1", "70,30"
	s := &Server{monitor: monitor.New(time.Hour)}

	err := s.validateGPUAssignment(
		&benchmark.ConfigOverrides{GPUAssign: &assign, TensorSplit: &split}, nil)
	if err == nil {
		t.Error("setting both GPU Assignment and Tensor Split must be refused")
	}

	// Sweeping one against a fixed other is the same contradiction.
	err = s.validateGPUAssignment(
		&benchmark.ConfigOverrides{TensorSplit: &split},
		[]benchmark.SweepAxis{{Field: "gpu_assign", Values: []string{"all", "0"}}})
	if err == nil {
		t.Error("sweeping gpu_assign against a fixed tensor_split must be refused")
	}
}

// A gpu_assign job needs a GPU count to resolve against; without one
// every value collapses to the same configuration.
func TestGPUAssignRefusedWithoutGPUCount(t *testing.T) {
	assign := "0-1"
	s := &Server{monitor: monitor.New(time.Hour)} // never polled: no GPUs
	if err := s.validateGPUAssignment(&benchmark.ConfigOverrides{GPUAssign: &assign}, nil); err == nil {
		t.Error("a gpu_assign job with no detected GPUs must be refused up front")
	}
}

// Jobs that don't touch GPU placement are unaffected.
func TestGPUPlacementValidationIgnoresUnrelatedJobs(t *testing.T) {
	s := &Server{monitor: monitor.New(time.Hour)}
	if err := s.validateGPUAssignment(nil, []benchmark.SweepAxis{
		{Field: "ubatch_size", Values: []string{"512", "1024"}},
	}); err != nil {
		t.Errorf("unrelated sweep refused: %v", err)
	}
}

// With no GPU count, ResolveGPUAssign returns empty for every value, so
// every sweep point would generate an identical preset and report
// identical measurements under different labels. Fail instead.
func TestGPUAssignFailsWhenGPUCountUnknown(t *testing.T) {
	base := models.ModelConfig{GPUAssign: "all"}
	out := applySnapshotToConfig(base, benchmark.ConfigSnapshot{GPUAssign: "0-1"})
	if err := resolveGPUAssignment(&out, base, 0); err == nil {
		t.Error("expected an error rather than a silently collapsed sweep")
	}
}

// The interactive guard reads two signals because each is blind alone:
// the queue's view goes false if a running job's record is deleted, and
// ownership covers that window. Ownership can no longer strand the
// guard, because cleanup releases it even when the restore fails.
func TestRouterBusyGuardUsesOwnershipWhenQueueIsBlind(t *testing.T) {
	s := &Server{}
	if s.routerBusyWithJob() {
		t.Error("nothing wired; nothing is busy")
	}

	// A job is driving the router even though the queue can't see it
	// (its record was deleted mid-run).
	s.env = &jobEnv{s: s, ownsRouter: true}
	if !s.routerBusyWithJob() {
		t.Error("a job still holding the router must keep the guards armed")
	}

	// Cleanup releases ownership unconditionally, so the guard clears
	// even when the restore restart failed — otherwise a single failure
	// locks the user out with no job left to cancel.
	s.env.releaseRouter()
	if s.routerBusyWithJob() {
		t.Error("released ownership must not block interactive start")
	}
}

// A job that changes nothing about the router must not restart it — not
// at the start, and not at the end either. Latching ownership on the
// no-restart fast path reinstated at teardown exactly the gratuitous
// restart the fast path removes, so the fix cancelled itself out.
func TestNoTakeoverMeansNoTeardownRestart(t *testing.T) {
	e := &jobEnv{}
	if e.routerOwnedByJob() {
		t.Fatal("nothing taken over yet")
	}

	// Simulate the fast path: build recorded, no configs, no restart.
	e.mu.Lock()
	e.jobBuildID = "build-A"
	e.mu.Unlock()

	if e.routerOwnedByJob() {
		t.Error("recording the build without restarting must not claim the router — cleanup would then restart it for nothing")
	}
}

// Speculative decoding must be fully controllable per job: a mode plus
// its parameters reach the launch flags, and the explicit off value
// ("none" in the form, empty in the snapshot) silences a model whose
// saved config has speculation enabled — the on-vs-off comparison in a
// single job.
func TestSpecParamsReachLaunchFlags(t *testing.T) {
	base := models.ModelConfig{Enabled: true, GPULayers: 999, ContextSize: 8192, Threads: 8}

	on := applySnapshotToConfig(base, benchmark.ConfigSnapshot{
		GPULayers: 999, ContextSize: 8192, Threads: 8,
		SpecType: "draft-mtp", DraftMax: 6, DraftPMin: "0.75",
	})
	flags := on.EffectiveFlagsFor(false, "")
	for _, want := range []string{"--spec-type draft-mtp", "--spec-draft-n-max 6", "--spec-draft-p-min 0.75"} {
		if !strings.Contains(flags, want) {
			t.Errorf("flags missing %q: %s", want, flags)
		}
	}

	savedMTP := base
	savedMTP.SpecType = "draft-mtp"
	savedMTP.DraftMax = 6
	off := applySnapshotToConfig(savedMTP, benchmark.ConfigSnapshot{
		GPULayers: 999, ContextSize: 8192, Threads: 8,
		// SpecType empty: the explicit off override.
	})
	if flags := off.EffectiveFlagsFor(false, ""); strings.Contains(flags, "--spec-type") {
		t.Errorf("explicit off should emit no speculative flags: %s", flags)
	}
}
