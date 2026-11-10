package models

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A projector kept in RAM is launched with --no-mmproj-offload (in the
// router's preset, mmproj-offload = false), and only when it is loaded.
func TestProjectorInRAMIsWrittenWhenItIsLoaded(t *testing.T) {
	tests := []struct {
		name string
		cfg  ModelConfig
		want bool
	}{
		{"on the GPU", ModelConfig{MmprojPath: "/models/mmproj.gguf"}, false},
		{"in RAM", ModelConfig{MmprojPath: "/models/mmproj.gguf", MmprojInRAM: true}, true},
		{"in RAM but not loaded", ModelConfig{MmprojPath: "/models/mmproj.gguf", MmprojDisabled: true, MmprojInRAM: true}, false},
		{"in RAM with no projector", ModelConfig{MmprojInRAM: true}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := tt.cfg
			cfg.Enabled = true
			target := Target{Backend: "cuda"}
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": &cfg}, target)
			flags := cfg.EffectiveFlagsFor(KindChat, target)
			if got := strings.Contains(ini, "mmproj-offload = false"); got != tt.want {
				t.Errorf("mmproj-offload = false in preset: %v, want %v\n%s", got, tt.want, ini)
			}
			if got := strings.Contains(flags, "--no-mmproj-offload"); got != tt.want {
				t.Errorf("--no-mmproj-offload in flags: %v, want %v: %s", got, tt.want, flags)
			}
		})
	}
}

// A projector in RAM leaves the GPU estimate and counts as system memory.
func TestProjectorInRAMMovesItToSystemMemory(t *testing.T) {
	p := filepath.Join(t.TempDir(), "mmproj.gguf")
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	f.Truncate(1 << 30)
	f.Close()

	onGPU := ModelConfig{MmprojPath: p}
	inRAM := ModelConfig{MmprojPath: p, MmprojInRAM: true}
	approx(t, "aux on GPU", AuxFilesVRAMGB(&onGPU), 1, 1e-9)
	approx(t, "aux in RAM", AuxFilesVRAMGB(&inRAM), 0, 1e-9)
	approx(t, "RAM on GPU", ProjectorRAMGB(&onGPU), 0, 1e-9)
	approx(t, "RAM in RAM", ProjectorRAMGB(&inRAM), 1, 1e-9)

	m := &Model{NLayers: 32, NEmbd: 4096, NHead: 32, NKVHead: 8, ContextLength: 32768, SizeBytes: 4 << 30}
	a := VRAMBreakdownForConfigOn(m, &onGPU, 1)
	b := VRAMBreakdownForConfigOn(m, &inRAM, 1)
	approx(t, "GPU total", a.Total()-b.Total(), 1, 1e-6)
	approx(t, "system memory", b.CPURAM-a.CPURAM, 1, 1e-6)
}
