package api

import (
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

func fitHardware(gpuGiB ...int) models.Hardware {
	hw := models.Hardware{LogicalCores: 16, RAMTotalMiB: 64 * 1024}
	for i, g := range gpuGiB {
		hw.GPUs = append(hw.GPUs, models.GPUSpec{Index: i, Name: "GPU", VRAMTotalMiB: g * 1024})
	}
	return hw
}

// An 8B-class dense model: 36 layers, 8 KV heads of 128.
func denseModel(sizeGiB int64, layers int) *models.Model {
	return &models.Model{
		SizeBytes: sizeGiB << 30, NLayers: layers, NEmbd: 4096, NHead: 32, NKVHead: 8,
		KVFullPerTok: layers * 8 * 256, ContextLength: 131072,
	}
}

func TestPlanFileFit(t *testing.T) {
	moe := denseModel(18, 48)
	moe.ExpertCount, moe.ExpertUsedCount, moe.ExpertLayers, moe.ExpertBytes = 128, 8, 48, 16<<30

	for _, tt := range []struct {
		name      string
		m         *models.Model
		hw        models.Hardware
		kind      string
		label     string
		inTooltip string
	}{
		{"small dense model", denseModel(5, 36), fitHardware(24), "gpu", "Up to 128K", "128K: all on the GPU, 8-bit KV cache"},
		{"dense model too large for the card", denseModel(40, 80), fitHardware(24), "partial", "Partly on CPU", "layers on the GPU"},
		{"MoE model larger than the card", moe, fitHardware(16), "experts", "Experts in RAM · up to 64K", "128K: experts of 33 layers in system memory, 8-bit KV cache, only 64K fits"},
		{"larger than the whole machine", denseModel(400, 120), fitHardware(24), "none", "Too large", "does not fit"},
		{"two cards hold what one cannot", denseModel(30, 64), fitHardware(24, 24), "gpu", "Up to 64K", "32K: all on the GPU, full-precision KV cache"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			fit := planFileFit(tt.m, tt.hw)
			if fit.Kind != tt.kind || fit.Label != tt.label {
				t.Errorf("got %s %q, want %s %q\n%s", fit.Kind, fit.Label, tt.kind, tt.label, fit.Detail)
			}
			if !strings.Contains(fit.Detail, tt.inTooltip) {
				t.Errorf("tooltip lacks %q:\n%s", tt.inTooltip, fit.Detail)
			}
			if fit.VRAMGiB <= float64(tt.m.SizeBytes>>30)/2 {
				t.Errorf("VRAM estimate %.1f GiB is implausibly small", fit.VRAMGiB)
			}
		})
	}
}

// One line per distinct context size: a model trained for 32K is not
// listed at 128K, and its "max" line is not repeated.
func TestPlanFileFitStopsAtTheTrainedContext(t *testing.T) {
	m := denseModel(5, 36)
	m.ContextLength = 32768
	fit := planFileFit(m, fitHardware(24))
	if fit.Label != "Up to 32K" {
		t.Errorf("label %q, want Up to 32K", fit.Label)
	}
	if n := strings.Count(fit.Detail, "\n") + 1; n != 2 {
		t.Errorf("%d lines, want 2 (32K and 8K):\n%s", n, fit.Detail)
	}
}

func TestTokensLabel(t *testing.T) {
	for n, want := range map[int]string{8192: "8K", 32768: "32K", 40960: "40K", 131072: "128K", 262144: "256K", 1 << 20: "1M", 500: "500"} {
		if got := tokensLabel(n); got != want {
			t.Errorf("tokensLabel(%d) = %q, want %q", n, got, want)
		}
	}
}

func TestMetaProbeFileIsTheLargestModelFile(t *testing.T) {
	files := []modelsource.File{
		{Filename: "mmproj-F16.gguf", Size: 30 << 30, IsMMProj: true},
		{Filename: "Model-Q4_K_M.gguf", Size: 5 << 30},
		{Filename: "Model-Q8_0.gguf", Size: 9 << 30},
		{Filename: "Draft-0.6B-Q8_0.gguf", Size: 1 << 29},
	}
	f, ok := metaProbeFile(files)
	if !ok || f.Filename != "Model-Q8_0.gguf" {
		t.Errorf("picked %q", f.Filename)
	}
	if _, ok := metaProbeFile(files[:1]); ok {
		t.Error("an mmproj-only listing yielded a probe file")
	}
}

// A draft model kept beside the main one must not be described by the
// main model's parameter count.
func TestSameModel(t *testing.T) {
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
		if got := sameModel(modelsource.File{Size: tt.size}, params); got != tt.want {
			t.Errorf("size %d: sameModel = %v, want %v", tt.size, got, tt.want)
		}
	}
	if !sameModel(modelsource.File{Size: 1}, 0) {
		t.Error("without a parameter count every file must count as the model")
	}
}

func TestFitCellRendersThePlan(t *testing.T) {
	view := hfModelView{ID: "org/Model-GGUF", Files: []hfFileView{
		{ModelFile: modelsource.File{Filename: "a.gguf", Size: 5 << 30}, Fit: &fileFit{
			Kind: "gpu", Label: "Up to 128K", Detail: "128K: all on the GPU, 8-bit KV cache", VRAMGiB: 14.2}},
		{ModelFile: modelsource.File{Filename: "b.gguf", Size: 18 << 30}, Fit: &fileFit{
			Kind: "experts", Label: "Experts in RAM · up to 32K", Detail: "32K: experts of 12 layers in system memory", VRAMGiB: 20.5}},
	}}
	out := renderFiles(t, "hf_file_estimates", view)
	for _, want := range []string{"Up to 128K", "14.2 GiB", "8-bit KV cache", "Experts in RAM", "generation is slower", "Real use can differ"} {
		if !strings.Contains(out, want) {
			t.Errorf("fill lacks %q; output=\n%s", want, out)
		}
	}
}

// An image reader is not a model: it gets no fit label, plan or fallback.
func TestFitCellForAnImageReader(t *testing.T) {
	view := hfModelView{ID: "org/Model-GGUF", Files: []hfFileView{
		{ModelFile: modelsource.File{Filename: "mmproj-F16.gguf", Size: 1 << 30, IsMMProj: true, VRAMEstGB: 1.1}},
	}}
	out := renderFiles(t, "hf_file_estimates", view)
	if strings.Contains(out, "Fits") || !strings.Contains(out, "image reader") {
		t.Errorf("image reader cell: %s", out)
	}
}
