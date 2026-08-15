package config

import (
	"strings"
	"testing"
)

func TestValidateRuntimeEnvAcceptsKnownValues(t *testing.T) {
	err := ValidateRuntimeEnv(map[string]string{
		"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f16",
		"ROCBLAS_USE_HIPBLASLT":         "1",
	})
	if err != nil {
		t.Errorf("valid env rejected: %v", err)
	}
}

// The curated list is the point: a typo or a variable from a forum post
// must be refused rather than silently set on llama-server.
func TestValidateRuntimeEnvRejectsUnknownName(t *testing.T) {
	err := ValidateRuntimeEnv(map[string]string{"HSA_NO_SCRATCH_RECLAIM": "1"})
	if err == nil {
		t.Error("expected an unknown variable to be rejected")
	}
}

func TestValidateRuntimeEnvRejectsBadValue(t *testing.T) {
	err := ValidateRuntimeEnv(map[string]string{"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f8"})
	if err == nil {
		t.Fatal("expected an invalid value to be rejected")
	}
	if !strings.Contains(err.Error(), "f16") {
		t.Errorf("error should list the valid values, got: %v", err)
	}
}

func TestValidateRuntimeEnvAllowsEmptyMeaningUnset(t *testing.T) {
	if err := ValidateRuntimeEnv(map[string]string{"ROCBLAS_USE_HIPBLASLT": ""}); err != nil {
		t.Errorf("empty value should be allowed as unset: %v", err)
	}
}

// Blank values must not reach the child process as KEY= — an unset
// option means "don't touch the environment".
func TestRuntimeEnvPairsSkipsBlanks(t *testing.T) {
	c := &Config{RuntimeEnv: map[string]string{
		"ROCBLAS_USE_HIPBLASLT":         "1",
		"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "",
		"GGML_CUDA_DISABLE_GRAPHS":      "  ",
	}}
	got := c.RuntimeEnvPairs()
	if len(got) != 1 || got[0] != "ROCBLAS_USE_HIPBLASLT=1" {
		t.Errorf("got %v, want only the set variable", got)
	}
}

func TestRuntimeEnvPairsSorted(t *testing.T) {
	c := &Config{RuntimeEnv: map[string]string{
		"ROCBLAS_USE_HIPBLASLT":         "1",
		"GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f16",
	}}
	got := c.RuntimeEnvPairs()
	if len(got) != 2 || got[0] != "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16" {
		t.Errorf("pairs should be sorted for a stable launch, got %v", got)
	}
}

func TestRuntimeEnvPairsEmpty(t *testing.T) {
	c := &Config{}
	if got := c.RuntimeEnvPairs(); got != nil {
		t.Errorf("got %v, want nil", got)
	}
}

// Free-form entries extend the environment and override curated values
// on a name clash — the escape hatch wins.
func TestEnvSetExtraOverridesCurated(t *testing.T) {
	e := EnvSet{
		Curated: map[string]string{"GGML_CUDA_DISABLE_GRAPHS": "1"},
		Extra:   "FOO=bar\n# a comment\n\nGGML_CUDA_DISABLE_GRAPHS=0",
	}
	got := e.Pairs()
	want := []string{"FOO=bar", "GGML_CUDA_DISABLE_GRAPHS=0"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("got %v, want %v", got, want)
	}
}

// Unknown names are the point of the free-form entry — allowed. Only
// malformed lines (not KEY=VALUE) are refused.
func TestEnvSetValidateExtra(t *testing.T) {
	if err := (EnvSet{Extra: "ANY_UNKNOWN_VAR=1"}).Validate(); err != nil {
		t.Errorf("unknown free-form variable should be allowed: %v", err)
	}
	if err := (EnvSet{Extra: "not a pair"}).Validate(); err == nil {
		t.Error("malformed line should be refused")
	}
	if err := (EnvSet{Extra: "1BAD=x"}).Validate(); err == nil {
		t.Error("invalid variable name should be refused")
	}
}

// Footguns warn but never block: the variables still validate, still
// apply, and each warning names the specific hazard.
func TestEnvSetWarnsOnFootguns(t *testing.T) {
	e := EnvSet{Extra: "HSA_OVERRIDE_GFX_VERSION=11.0.0\nCUDA_DEVICE_ORDER=FASTEST_FIRST\nSAFE_VAR=1"}
	if err := e.Validate(); err != nil {
		t.Fatalf("footguns must save, not block: %v", err)
	}
	warnings := e.Warnings()
	if len(warnings) != 2 {
		t.Fatalf("want 2 warnings, got %v", warnings)
	}
	if !strings.Contains(warnings[0], "CUDA_DEVICE_ORDER") && !strings.Contains(warnings[1], "CUDA_DEVICE_ORDER") {
		t.Errorf("expected a CUDA_DEVICE_ORDER warning, got %v", warnings)
	}
	pairs := e.Pairs()
	if len(pairs) != 3 {
		t.Errorf("footgun variables must still apply, got %v", pairs)
	}
}

// The component is scope-agnostic: it operates on plain data with no
// dependency on the global Config, so a future per-model EnvSet reuses
// it unchanged. This test IS that demonstration — a standalone set,
// validated, warned, and rendered without a Config in sight.
func TestEnvSetUsableAtNonGlobalScope(t *testing.T) {
	perModel := EnvSet{
		Curated: map[string]string{"GGML_VK_DISABLE_COOPMAT": "1"},
		Extra:   "MODEL_SPECIFIC=yes",
	}
	if err := perModel.Validate(); err != nil {
		t.Fatalf("standalone set should validate: %v", err)
	}
	if got := perModel.Pairs(); len(got) != 2 {
		t.Errorf("standalone set should render pairs, got %v", got)
	}
}

// Backend filtering keeps the table short, but a variable with a value
// set must stay visible even when the backend doesn't match — otherwise
// switching builds leaves it applying invisibly with no way to clear it.
func TestFilterRuntimeEnvOptions(t *testing.T) {
	rocm := FilterRuntimeEnvOptions("rocm", nil)
	for _, o := range rocm {
		for _, b := range o.Backends {
			if b == "vulkan" && len(o.Backends) == 1 {
				t.Errorf("vulkan-only option %s shown on rocm", o.Name)
			}
		}
	}
	// A set vulkan-only var must survive the rocm filter.
	set := map[string]string{"GGML_VK_DISABLE_COOPMAT": "1"}
	found := false
	for _, o := range FilterRuntimeEnvOptions("rocm", set) {
		if o.Name == "GGML_VK_DISABLE_COOPMAT" {
			found = true
		}
	}
	if !found {
		t.Error("a set variable must stay visible regardless of backend filter")
	}
	// Unknown backend shows everything.
	if got, all := len(FilterRuntimeEnvOptions("", nil)), len(RuntimeEnvOptions()); got != all {
		t.Errorf("empty backend should show all options, got %d of %d", got, all)
	}
}

// The variables we deliberately excluded are the ones most likely to be
// pasted in from forum threads; keep that decision pinned.
func TestRuntimeEnvOptionsExcludeUnsupportedFolklore(t *testing.T) {
	for _, opt := range RuntimeEnvOptions() {
		switch opt.Name {
		case "HSA_NO_SCRATCH_RECLAIM", "HSA_ENABLE_SDMA", "HSA_OVERRIDE_GFX_VERSION":
			t.Errorf("%s should not be offered: no llama.cpp evidence, or harmful on supported GPUs", opt.Name)
		case "GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F":
			t.Errorf("%s is a build flag removed upstream, not a runtime variable", opt.Name)
		}
	}
}
