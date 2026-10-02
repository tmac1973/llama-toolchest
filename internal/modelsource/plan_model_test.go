package modelsource

import (
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func TestMetaProbeFileIsTheLargestModelFile(t *testing.T) {
	files := []File{
		{Filename: "mmproj-F16.gguf", Size: 30 << 30, IsMMProj: true},
		{Filename: "Model-Q4_K_M.gguf", Size: 5 << 30},
		{Filename: "Model-Q8_0.gguf", Size: 9 << 30},
		{Filename: "Draft-0.6B-Q8_0.gguf", Size: 1 << 29},
	}
	f, ok := MetaProbeFile(files)
	if !ok || f.Filename != "Model-Q8_0.gguf" {
		t.Errorf("picked %q", f.Filename)
	}
	if _, ok := MetaProbeFile(files[:1]); ok {
		t.Error("an mmproj-only listing yielded a probe file")
	}
}

// A draft model kept beside the main one must not be described by the
// main model's parameter count.
func TestPlausibleFile(t *testing.T) {
	const params = 8_000_000_000
	for _, tt := range []struct {
		size int64
		want bool
	}{
		{5 << 30, true},      // Q4-class: about 5.4 bits per weight
		{16 << 30, true},     // F16
		{700 << 20, false},   // a 0.6B draft model at Q8
		{9 << 30, true},      // Q8_0
		{int64(2.4e9), true}, // an IQ2 quant: 2.4 bits per weight
		{int64(4e10), false}, // far above F32: a different model
	} {
		if got := PlausibleFile(File{Size: tt.size}, params); got != tt.want {
			t.Errorf("size %d: PlausibleFile = %v, want %v", tt.size, got, tt.want)
		}
	}
	if !PlausibleFile(File{Size: 1}, 0) {
		t.Error("without a parameter count every file must count as the model")
	}
}

func TestParamsFromFile(t *testing.T) {
	if got := ParamsFromFile(File{Filename: "m-Q8_0.gguf", Quant: "Q8_0", Size: 8_500_000_000}); got != 8e9 {
		t.Errorf("Q8_0 = %d", got)
	}
	if got := ParamsFromFile(File{Filename: "m-weird.gguf", Quant: "unknown", Size: 1}); got != 0 {
		t.Errorf("unknown quant = %d", got)
	}
}

// HuggingFace's count for peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP
// described its image reader (0.45B). The largest file says otherwise.
func TestRepoParamsDistrustsAnImplausibleCount(t *testing.T) {
	files := []File{
		{Filename: "Tiel-Coder-35B-A3B-MTP-UD-Q5_K_XL.gguf", Quant: "UD_Q5_K_XL", Size: 26_981_932_896},
		{Filename: "Tiel-Coder-35B-A3B-MTP-UD-Q8_K_XL.gguf", Quant: "UD_Q8_K_XL", Size: 38_840_606_560},
		{Filename: "mmproj-BF16.gguf", Size: 902_822_016, IsMMProj: true},
	}
	got := RepoParams(files, 446_571_248)
	if got < 34e9 || got > 38e9 {
		t.Errorf("RepoParams = %d, want about 36.5B", got)
	}
	if got := RepoParams(files, 36_555_864_997); got != 36_555_864_997 {
		t.Errorf("a plausible count was replaced: %d", got)
	}
}

// A measured per-layer embedding table replaces the derived one: llama.cpp
// keeps it in system memory, so it must not count as GPU memory.
func TestPlanModelUsesTheMeasuredEmbeddingTable(t *testing.T) {
	meta := &models.GGUFMeta{Architecture: "qwen4exp", NLayers: 48, NEmbd: 4096, NHead: 32, MetaOnly: true}
	f := File{Filename: "m.gguf", Size: 80 << 30, StreamProbed: true, StreamedBytes: 26 << 30}
	if m := PlanModel(meta, f, 170e9); m.PLEBytes != 26<<30 {
		t.Errorf("PLEBytes = %d, want the measured 26 GiB", m.PLEBytes)
	}
}
