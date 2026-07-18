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
