package modelsource

import "testing"

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
