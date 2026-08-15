package models

import (
	"strings"
	"testing"
)

func TestSpecMTPEffectiveFlags(t *testing.T) {
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		DraftMax:    6,
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{"--spec-type draft-mtp", "--spec-draft-n-max 6"} {
		if !strings.Contains(got, want) {
			t.Errorf("MTP flags missing %q in: %s", want, got)
		}
	}
	if strings.Contains(got, "--model-draft") {
		t.Errorf("MTP should not emit --model-draft, got: %s", got)
	}
}

func TestSpecMTPSeparateDrafterFlags(t *testing.T) {
	// gemma-4 style: MTP head ships as its own GGUF, loaded via --model-draft
	// under spec-type draft-mtp, including draft-resource overrides.
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		MtpPath:     "/models/gemma-4-12B-it-MTP-Q8_0.gguf",
		DraftMax:    4,
		DraftDevice: "CUDA0",
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{
		"--spec-type draft-mtp",
		"--model-draft /models/gemma-4-12B-it-MTP-Q8_0.gguf",
		"--spec-draft-n-max 4",
		"--device-draft CUDA0",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("gemma MTP flags missing %q in: %s", want, got)
		}
	}
}

func TestSpecMTPSeparateDrafterDisabled(t *testing.T) {
	// MtpDisabled suppresses --model-draft while keeping self-speculation flags.
	cfg := &ModelConfig{
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		MtpPath:     "/models/gemma-4-12B-it-MTP-Q8_0.gguf",
		MtpDisabled: true,
		DraftMax:    4,
	}
	got := cfg.EffectiveFlags()
	if strings.Contains(got, "--model-draft") {
		t.Errorf("disabled MTP head should not emit --model-draft, got: %s", got)
	}
	if !strings.Contains(got, "--spec-type draft-mtp") {
		t.Errorf("MTP flags missing --spec-type draft-mtp in: %s", got)
	}
}

func TestIsMTPHeadArch(t *testing.T) {
	for _, arch := range []string{"gemma4-assistant", "gemma4_assistant"} {
		if !IsMTPHeadArch(arch) {
			t.Errorf("IsMTPHeadArch(%q) = false, want true", arch)
		}
	}
	// Qwen self-speculation MTP models use a normal runnable arch — must NOT
	// be classified as a standalone drafter head.
	for _, arch := range []string{"qwen3", "qwen3moe", "gemma3", "llama"} {
		if IsMTPHeadArch(arch) {
			t.Errorf("IsMTPHeadArch(%q) = true, want false", arch)
		}
	}
}

func TestSpecDraftResourceFlags(t *testing.T) {
	cfg := &ModelConfig{
		GPULayers:         99,
		ContextSize:       16384,
		Threads:           8,
		SpecType:          "draft",
		DraftModelPath:    "/models/qwen-0.5b.gguf",
		DraftMax:          16,
		DraftPMin:         "0.75",
		DraftCtxSize:      4096,
		DraftGPULayers:    99,
		DraftDevice:       "CUDA1",
		DraftCPUMoE:       2,
		DraftKVCacheQuant: "q8_0",
	}
	got := cfg.EffectiveFlags()
	for _, want := range []string{
		"--spec-type draft-simple",
		"--model-draft /models/qwen-0.5b.gguf",
		"--spec-draft-n-max 16",
		"--spec-draft-p-min 0.75",
		"--ctx-size-draft 4096",
		"--gpu-layers-draft 99",
		"--device-draft CUDA1",
		"--n-cpu-moe-draft 2",
		"--cache-type-k-draft q8_0",
		"--cache-type-v-draft q8_0",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("Draft flags missing %q in: %s", want, got)
		}
	}
}

func TestSpecMTPPresetINI(t *testing.T) {
	cfg := &ModelConfig{
		Enabled:     true,
		GPULayers:   99,
		ContextSize: 8192,
		Threads:     8,
		SpecType:    "draft-mtp",
		DraftMax:    6,
	}
	var b strings.Builder
	writeConfigParams(&b, cfg, false, "")
	out := b.String()
	for _, want := range []string{"spec-type = draft-mtp", "spec-draft-n-max = 6"} {
		if !strings.Contains(out, want) {
			t.Errorf("MTP preset missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "model-draft") {
		t.Errorf("MTP preset should not emit model-draft, got:\n%s", out)
	}
}
