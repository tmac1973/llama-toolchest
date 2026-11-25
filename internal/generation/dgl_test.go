package generation

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var updateGolden = flag.Bool("update-dgl", false, "rewrite the golden DGL scripts")

func dglTestSet(script string, scale int) ModelSet {
	return ModelSet{ID: "s", Kind: KindVUpscale, Recipe: script, Scale: scale, Components: []Component{
		{Slot: "dit", Path: "/models/gen-models/" + script + "/dit.safetensors"},
		{Slot: "vae", Path: "/models/gen-models/" + script + "/vae.safetensors"},
	}}
}

func TestRenderVUpscaleGolden(t *testing.T) {
	cases := []struct {
		name   string
		set    ModelSet
		format string
		window int
		fps    int
	}{
		{"relayvsr-webm", dglTestSet("relayvsr", 2), "webm", 16, 0},
		{"relayvsr-avi", dglTestSet("relayvsr", 4), "avi", 8, 0},
		{"seedvr2-webm", dglTestSet("seedvr2", 2), "webm", 1, 24},
		{"seedvr2-avi-x3", dglTestSet("seedvr2", 3), "avi", 5, 0},
	}
	for _, c := range cases {
		got, err := RenderVUpscale(c.set, VUpscaleJob{
			InputPath: "input/job1.webm", OutputPath: "output/job1." + c.format,
			OutputFormat: c.format, Seed: 42, WindowSize: c.window, FPS: c.fps,
		})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		path := filepath.Join("testdata", "dgl", c.name+".golden.dgl")
		if *updateGolden {
			if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		want, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("%s: %v (run go test -run TestRenderVUpscaleGolden -update-dgl)", c.name, err)
		}
		if got != string(want) {
			t.Errorf("%s: script differs from %s (run with -update-dgl after checking)", c.name, path)
		}
	}
}

func TestRenderVUpscaleRefuses(t *testing.T) {
	job := VUpscaleJob{InputPath: "input/a.avi", OutputPath: "output/a.webm", OutputFormat: "webm", WindowSize: 16}
	bad := dglTestSet("relayvsr", 2)
	bad.Components[0].Path = `/models/my "best" dit.safetensors`
	sv := dglTestSet("seedvr2", 2)
	wan := dglTestSet("relayvsr", 2)
	wan.Recipe = "wan"
	tests := []struct {
		name string
		set  ModelSet
		job  VUpscaleJob
		want string
	}{
		{"quote", bad, job, "has a double quote"},
		{"seedvr2 window 4", sv, VUpscaleJob{InputPath: "i", OutputPath: "o", OutputFormat: "webm", WindowSize: 4}, "window_size must be 1, 5, 9"},
		{"relayvsr window 1", dglTestSet("relayvsr", 2), VUpscaleJob{InputPath: "i", OutputPath: "o", OutputFormat: "webm", WindowSize: 1}, "between 2 and 32"},
		{"script", wan, job, "unknown video upscaler script"},
		{"format", dglTestSet("relayvsr", 2), VUpscaleJob{InputPath: "i", OutputPath: "o", OutputFormat: "mp4", WindowSize: 16}, "webm or avi"},
		{"newline", dglTestSet("relayvsr", 2), VUpscaleJob{InputPath: "i\nmain()", OutputPath: "o", OutputFormat: "webm", WindowSize: 16}, "control character"},
	}
	for _, tt := range tests {
		if _, err := RenderVUpscale(tt.set, tt.job); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: err = %v, want %q", tt.name, err, tt.want)
		}
	}
}

func TestWindowSizes(t *testing.T) {
	if DefaultWindowSize("relayvsr") != 16 || DefaultWindowSize("seedvr2") != 1 {
		t.Error("defaults")
	}
	for _, n := range []int{1, 5, 9, 33} {
		if err := WindowSizeOK("seedvr2", n); err != nil {
			t.Errorf("seedvr2 %d: %v", n, err)
		}
	}
	for _, n := range []int{2, 16, 32} {
		if err := WindowSizeOK("relayvsr", n); err != nil {
			t.Errorf("relayvsr %d: %v", n, err)
		}
	}
}

func esrganTestSet() ModelSet {
	return ModelSet{ID: "esrgan", Kind: KindUpscale, Components: []Component{
		{Slot: UpscalerSlot, Path: "/models/gen-models/Comfy-Org--Real-ESRGAN/RealESRGAN_x4plus.safetensors"},
	}}
}

func TestRenderUpscaleGolden(t *testing.T) {
	got, err := RenderUpscale(esrganTestSet(), UpscaleJob{InputPath: "input/job1.png", OutputPath: "output/job1.png", Repeats: 2})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join("testdata", "dgl", "esrgan.golden.dgl")
	if *updateGolden {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run go test -run TestRenderUpscaleGolden -update-dgl)", err)
	}
	if got != string(want) {
		t.Errorf("script differs from %s (run with -update-dgl after checking)", path)
	}
}

func TestRenderUpscaleRefuses(t *testing.T) {
	ok := UpscaleJob{InputPath: "input/a.png", OutputPath: "output/a.png", Repeats: 1}
	cases := []struct {
		name string
		set  ModelSet
		job  func(UpscaleJob) UpscaleJob
		want string
	}{
		{"wrong kind", ModelSet{ID: "v", Kind: KindVUpscale}, func(j UpscaleJob) UpscaleJob { return j }, "not an upscaler set"},
		{"no file", ModelSet{ID: "e", Kind: KindUpscale}, func(j UpscaleJob) UpscaleJob { return j }, "the upscaler's path"},
		{"repeats 0", esrganTestSet(), func(j UpscaleJob) UpscaleJob { j.Repeats = 0; return j }, "repeats"},
		{"repeats 5", esrganTestSet(), func(j UpscaleJob) UpscaleJob { j.Repeats = 5; return j }, "repeats"},
		{"small tile", esrganTestSet(), func(j UpscaleJob) UpscaleJob { j.TileSize = 16; return j }, "tile_size"},
		{"not png", esrganTestSet(), func(j UpscaleJob) UpscaleJob { j.OutputPath = "output/a.jpg"; return j }, ".png"},
		{"quote", esrganTestSet(), func(j UpscaleJob) UpscaleJob { j.InputPath = `input/a".png`; return j }, "double quote"},
	}
	for _, c := range cases {
		_, err := RenderUpscale(c.set, c.job(ok))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want one with %q", c.name, err, c.want)
		}
	}
	src, err := RenderUpscale(esrganTestSet(), UpscaleJob{InputPath: "input/a.png", OutputPath: "output/a.png", Repeats: 1, TileSize: 256})
	if err != nil || !strings.Contains(src, "tile_size = 256") {
		t.Errorf("tile 256: err %v, script:\n%s", err, src)
	}
}
