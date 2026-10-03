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
		kind      models.Placement
		label     string
		inTooltip string
	}{
		{"small dense model", denseModel(5, 36), fitHardware(24), "gpu", "Up to 128K", "128K: all on the GPU, 8-bit KV cache"},
		{"dense model too large for the card", denseModel(40, 80), fitHardware(24), "partial", "Partly on CPU", "layers on the GPU"},
		{"MoE model larger than the card", moe, fitHardware(16), "experts", "Experts in RAM · up to 64K", "128K: experts of 32 layers in system memory, 8-bit KV cache, only 64K fits"},
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

func TestFitCellRendersThePlan(t *testing.T) {
	view := hfModelView{ID: "org/Model-GGUF", Files: []hfFileView{
		{ModelFile: modelsource.File{Filename: "a.gguf", Size: 5 << 30}, Fit: &fileFit{
			Kind: models.PlacementGPU, Label: "Up to 128K", Detail: "128K: all on the GPU, 8-bit KV cache", VRAMGiB: 14.2}},
		{ModelFile: modelsource.File{Filename: "b.gguf", Size: 18 << 30}, Fit: &fileFit{
			Kind: models.PlacementExperts, Label: "Experts in RAM · up to 32K", Detail: "32K: experts of 12 layers in system memory", VRAMGiB: 20.5}},
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

// The size-only label, used when a file cannot be planned, counts the
// same cards the planner does. A 16 GB card with a 2 GB integrated GPU
// read as room for "2 GPU".
func TestSizeOnlyLabelIgnoresTheIntegratedGPU(t *testing.T) {
	hw := models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, Name: "RX 9070 XT", VRAMTotalMiB: 16 * 1024},
		{Index: 1, Name: "iGPU", VRAMTotalMiB: 2 * 1024, IsIGPU: true},
	}}
	s := &Server{testHardware: &hw}
	vramFit := s.templateFuncs()["vramFit"].(func(float64) string)
	if got := vramFit(23.3); got != "too_large" {
		t.Errorf("23.3 GiB on one 16 GB card = %q, want too_large", got)
	}
	if got := vramFit(10); got != "fits" {
		t.Errorf("10 GiB = %q, want fits", got)
	}
}

// The VRAM column shows the plan the Fit column describes. For a model
// that keeps experts in system memory, that is what fits on the card, with
// the system-memory part beside it, not the every-layer-on-the-GPU figure.
func TestVRAMColumnFollowsThePlan(t *testing.T) {
	moe := denseModel(18, 48)
	moe.ExpertCount, moe.ExpertUsedCount, moe.ExpertLayers, moe.ExpertBytes = 128, 8, 48, 16<<30
	hw := fitHardware(16)
	fit := planFileFit(moe, hw)
	budget, _ := models.FitBudgets(hw)
	if fit.Kind != models.PlacementExperts || fit.VRAMGiB > budget || fit.RAMGiB <= 0 || fit.RAMKind != models.PlacementExperts {
		t.Errorf("fit = %+v, budget %.1f GiB", fit, budget)
	}

	view := hfModelView{ID: "org/M-GGUF", Files: []hfFileView{{ModelFile: modelsource.File{Filename: "m.gguf", Size: 18 << 30}, Fit: fit}}}
	out := renderFiles(t, "hf_file_estimates", view)
	if !strings.Contains(out, "of experts in system memory") {
		t.Errorf("no system-memory line:\n%s", out)
	}

	// Too large for anything: the every-layer figure, no system-memory line.
	huge := planFileFit(denseModel(400, 120), hw)
	if huge.RAMGiB != 0 || huge.VRAMGiB < 400 {
		t.Errorf("too large: %+v", huge)
	}
}

// The embedding-table line says what happens to the table: llama.cpp
// reads a large one from disk as needed and loads a small one into system
// memory. Beside "of experts in system memory" a bare "held in system
// memory" read as a second, unexplained amount of RAM.
func TestEmbeddingTableLine(t *testing.T) {
	for _, tt := range []struct {
		bytes        int64
		want, unwant string
	}{
		{27 << 30, "27.0 GiB embedding table, read from disk", "table in system memory"},
		{2 << 30, "2.0 GiB embedding table in system memory", "read from disk"},
	} {
		view := hfModelView{ID: "org/M-GGUF", Files: []hfFileView{{ModelFile: modelsource.File{
			Filename: "m.gguf", Size: 60 << 30, StreamedBytes: tt.bytes, StreamProbed: true, VRAMEstGB: 30}}}}
		out := renderFiles(t, "hf_file_estimates", view)
		if !strings.Contains(out, tt.want) || strings.Contains(out, tt.unwant) {
			t.Errorf("%d GiB table:\n%s", tt.bytes>>30, out)
		}
	}
}
