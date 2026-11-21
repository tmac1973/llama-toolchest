package builder

import (
	"os"
	"path/filepath"
	"testing"
)

func testBuilder(t *testing.T) *Builder {
	t.Helper()
	return NewBuilder(t.TempDir())
}

// Save/load/delete round trip, including the upsert-on-same-name flow
// and profile scoping of the listing.
func TestFlagPresetCRUD(t *testing.T) {
	b := testBuilder(t)

	p := FlagPreset{
		Name:    "rocwmma",
		Profile: "rocm",
		Options: map[string]bool{
			"GGML_HIP_ROCWMMA_FATTN": true,
			"GGML_CUDA_FORCE_MMQ":    false,
		},
		ExtraCMake: "-DFOO=BAR",
	}
	if err := b.SaveFlagPreset(p); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := b.SaveFlagPreset(FlagPreset{Name: "fast", Profile: "cuda", Options: map[string]bool{}}); err != nil {
		t.Fatalf("save cuda: %v", err)
	}

	// Profile scoping: rocm listing must not leak the cuda preset.
	rocm := b.FlagPresets("", "rocm")
	if len(rocm) != 1 || rocm[0].Name != "rocwmma" {
		t.Fatalf("rocm listing = %+v", rocm)
	}
	if got := b.FlagPresets("", ""); len(got) != 2 {
		t.Fatalf("all listing = %+v", got)
	}

	// Upsert: same name replaces, no duplicate.
	p.ExtraCMake = "-DBAZ=1"
	if err := b.SaveFlagPreset(p); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	got, ok := b.FindFlagPreset(ProjectLlamaCpp, "rocwmma")
	if !ok || got.ExtraCMake != "-DBAZ=1" || len(b.FlagPresets("", "rocm")) != 1 {
		t.Fatalf("upsert result: %+v", got)
	}
	// Off-toggles are stored explicitly so applying doesn't fall back to
	// defaults for them.
	if v, present := got.Options["GGML_CUDA_FORCE_MMQ"]; !present || v {
		t.Errorf("off toggle should be stored as explicit false, got %v/%v", v, present)
	}

	// Persistence across Builder instances (same dataDir).
	b2 := NewBuilder(b.dataDir)
	if _, ok := b2.FindFlagPreset(ProjectLlamaCpp, "rocwmma"); !ok {
		t.Error("preset did not persist to disk")
	}

	if found, err := b.DeleteFlagPreset(ProjectLlamaCpp, "rocwmma"); err != nil || !found {
		t.Error("delete reported missing")
	}
	if _, ok := b.FindFlagPreset(ProjectLlamaCpp, "rocwmma"); ok {
		t.Error("preset survived delete")
	}
}

// Names follow the build-tag rules — the preset name doubles as the tag
// labeling builds made from it.
func TestFlagPresetNameValidation(t *testing.T) {
	b := testBuilder(t)
	for _, bad := range []string{"", "Has Spaces", "UPPER", "-leading"} {
		if err := b.SaveFlagPreset(FlagPreset{Name: bad, Profile: "rocm"}); err == nil {
			t.Errorf("name %q should be rejected", bad)
		}
	}
	if err := b.SaveFlagPreset(FlagPreset{Name: "ok-name-2", Profile: "rocm"}); err != nil {
		t.Errorf("valid name rejected: %v", err)
	}
	if err := b.SaveFlagPreset(FlagPreset{Name: "no-profile"}); err == nil {
		t.Error("missing profile should be rejected")
	}
}

// A CUDA preset saved before FlagPresetVersion 1 built with graphs on
// whatever its toggle said, so it is updated to on: stored false (the
// old default) and a missing toggle alike. The update runs once.
func TestFlagPresetMigratesCUDAGraphsOnce(t *testing.T) {
	dir := t.TempDir()
	old := `[
  {"name": "was-off", "profile": "cuda", "options": {"GGML_CUDA_GRAPHS": false, "GGML_CUDA_FORCE_MMQ": true}},
  {"name": "was-on", "profile": "cuda", "options": {"GGML_CUDA_GRAPHS": true}},
  {"name": "missing", "profile": "cuda", "options": {}},
  {"name": "rocm", "profile": "rocm", "options": {"GGML_CUDA_FORCE_MMQ": false}}
]`
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config", "build-flag-presets.json"), []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}

	b := NewBuilder(dir)
	for _, name := range []string{"was-off", "was-on", "missing"} {
		p, _ := b.FindFlagPreset(ProjectLlamaCpp, name)
		if !p.Options["GGML_CUDA_GRAPHS"] || p.Version != FlagPresetVersion {
			t.Errorf("%s: graphs=%v version=%d, want on at version %d", name, p.Options["GGML_CUDA_GRAPHS"], p.Version, FlagPresetVersion)
		}
	}
	if p, _ := b.FindFlagPreset(ProjectLlamaCpp, "was-off"); !p.Options["GGML_CUDA_FORCE_MMQ"] {
		t.Error("other toggles must be kept")
	}
	if p, _ := b.FindFlagPreset(ProjectLlamaCpp, "rocm"); len(p.Options) != 1 || p.Version != FlagPresetVersion {
		t.Errorf("rocm preset should only be stamped: %+v", p)
	}

	// Turned off under the new meaning, it stays off: on disk and when
	// read back by a new Builder.
	if err := b.SaveFlagPreset(FlagPreset{Name: "was-off", Project: ProjectLlamaCpp, Profile: "cuda", Options: map[string]bool{"GGML_CUDA_GRAPHS": false}, Version: FlagPresetVersion}); err != nil {
		t.Fatal(err)
	}
	if p, _ := NewBuilder(dir).FindFlagPreset(ProjectLlamaCpp, "was-off"); p.Options["GGML_CUDA_GRAPHS"] {
		t.Error("a preset saved at the current version was migrated again")
	}
}

// A preset restored from an older backup carries no Version and is
// migrated on save.
func TestSaveFlagPresetMigratesUnversioned(t *testing.T) {
	b := testBuilder(t)
	if err := b.SaveFlagPreset(FlagPreset{Name: "restored", Profile: "cuda", Options: map[string]bool{"GGML_CUDA_GRAPHS": false}}); err != nil {
		t.Fatal(err)
	}
	if p, _ := b.FindFlagPreset(ProjectLlamaCpp, "restored"); !p.Options["GGML_CUDA_GRAPHS"] {
		t.Error("an unversioned preset should be migrated on save")
	}
}

// A save that cannot write the file is reported to the caller, not
// dropped: here the config directory's place is taken by a plain file.
func TestSaveFlagPresetReportsWriteFailure(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	b := NewBuilder(dir)
	if err := b.SaveFlagPreset(FlagPreset{Name: "fast", Profile: "cuda", Options: map[string]bool{}}); err == nil {
		t.Fatal("save succeeded although the config directory cannot be created")
	}
}

// Sets belong to a project: the listing filters by it, the same name can
// exist in two projects, and find and delete touch only the named
// project's set.
func TestFlagPresetsPerProject(t *testing.T) {
	b := testBuilder(t)
	for _, p := range []FlagPreset{
		{Name: "fast", Project: ProjectLlamaCpp, Profile: "rocm", ExtraCMake: "-DLLAMA=1"},
		{Name: "fast", Project: ProjectSDCpp, Profile: "rocm", ExtraCMake: "-DSD=1"},
		{Name: "only-sd", Project: ProjectSDCpp, Profile: "rocm"},
	} {
		if err := b.SaveFlagPreset(p); err != nil {
			t.Fatalf("save %s/%s: %v", p.Project, p.Name, err)
		}
	}
	if got := b.FlagPresets(ProjectLlamaCpp, "rocm"); len(got) != 1 || got[0].ExtraCMake != "-DLLAMA=1" {
		t.Errorf("llama.cpp listing = %+v", got)
	}
	if got := b.FlagPresets(ProjectSDCpp, "rocm"); len(got) != 2 {
		t.Errorf("sd.cpp listing = %+v", got)
	}
	if got := b.FlagPresets(ProjectAceStep, ""); len(got) != 0 {
		t.Errorf("acestep listing = %+v; want none", got)
	}
	if got := b.FlagPresets("", ""); len(got) != 3 {
		t.Errorf("all listing = %+v", got)
	}
	if _, ok := b.FindFlagPreset(ProjectLlamaCpp, "only-sd"); ok {
		t.Error("found another project's set")
	}
	if p, ok := b.FindFlagPreset(ProjectSDCpp, "fast"); !ok || p.ExtraCMake != "-DSD=1" {
		t.Errorf("sd.cpp fast = %+v, %v", p, ok)
	}
	if found, err := b.DeleteFlagPreset(ProjectSDCpp, "fast"); err != nil || !found {
		t.Fatalf("delete: %v, %v", found, err)
	}
	if _, ok := b.FindFlagPreset(ProjectLlamaCpp, "fast"); !ok {
		t.Error("deleting the sd.cpp set removed the llama.cpp set of the same name")
	}
}

// Sets saved before they recorded a project become llama.cpp sets, on
// load and on a restore from an older backup. They were all saved with
// the llama.cpp toggle list.
func TestFlagPresetMigratesMissingProject(t *testing.T) {
	dir := t.TempDir()
	old := `[
  {"name": "v0", "profile": "rocm", "options": {}},
  {"name": "v1", "profile": "cuda", "options": {"GGML_CUDA_GRAPHS": false}, "version": 1}
]`
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config", "build-flag-presets.json"), []byte(old), 0o644); err != nil {
		t.Fatal(err)
	}
	b := NewBuilder(dir)
	for _, name := range []string{"v0", "v1"} {
		if p, ok := b.FindFlagPreset(ProjectLlamaCpp, name); !ok || p.Version != FlagPresetVersion {
			t.Errorf("%s: %+v, %v; want a llama.cpp set at version %d", name, p, ok, FlagPresetVersion)
		}
	}
	// The CUDA graphs step is for version 0 only: a version 1 off stays off.
	if p, _ := b.FindFlagPreset(ProjectLlamaCpp, "v1"); p.Options["GGML_CUDA_GRAPHS"] {
		t.Error("a version 1 set was given the version 0 CUDA graphs step")
	}
	if got := b.FlagPresets(ProjectSDCpp, ""); len(got) != 0 {
		t.Errorf("old sets must not show for sd.cpp: %+v", got)
	}

	if err := b.SaveFlagPreset(FlagPreset{Name: "restored", Profile: "cpu", Version: 1}); err != nil {
		t.Fatal(err)
	}
	if _, ok := b.FindFlagPreset(ProjectLlamaCpp, "restored"); !ok {
		t.Error("a restored set without a project should become a llama.cpp set")
	}
	if err := b.SaveFlagPreset(FlagPreset{Name: "no-project", Profile: "cpu", Version: FlagPresetVersion}); err == nil {
		t.Error("a current-version set without a project should be rejected")
	}
}
