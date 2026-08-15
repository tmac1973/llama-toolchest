package builder

import "testing"

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
	rocm := b.FlagPresets("rocm")
	if len(rocm) != 1 || rocm[0].Name != "rocwmma" {
		t.Fatalf("rocm listing = %+v", rocm)
	}
	if got := b.FlagPresets(""); len(got) != 2 {
		t.Fatalf("all listing = %+v", got)
	}

	// Upsert: same name replaces, no duplicate.
	p.ExtraCMake = "-DBAZ=1"
	if err := b.SaveFlagPreset(p); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	got, ok := b.FindFlagPreset("rocwmma")
	if !ok || got.ExtraCMake != "-DBAZ=1" || len(b.FlagPresets("rocm")) != 1 {
		t.Fatalf("upsert result: %+v", got)
	}
	// Off-toggles are stored explicitly so applying doesn't fall back to
	// defaults for them.
	if v, present := got.Options["GGML_CUDA_FORCE_MMQ"]; !present || v {
		t.Errorf("off toggle should be stored as explicit false, got %v/%v", v, present)
	}

	// Persistence across Builder instances (same dataDir).
	b2 := NewBuilder(b.dataDir)
	if _, ok := b2.FindFlagPreset("rocwmma"); !ok {
		t.Error("preset did not persist to disk")
	}

	if !b.DeleteFlagPreset("rocwmma") {
		t.Error("delete reported missing")
	}
	if _, ok := b.FindFlagPreset("rocwmma"); ok {
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
