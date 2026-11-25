package generation

import (
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"testing"
)

func TestValidateVUpscaleComponents(t *testing.T) {
	dit := Component{Slot: "dit", Path: "/m/dit.safetensors"}
	vae := Component{Slot: "vae", Path: "/m/vae.safetensors"}
	for _, script := range VUpscaleScripts {
		if err := ValidateComponents(KindVUpscale, script, []Component{dit, vae}); err != nil {
			t.Errorf("%s: %v", script, err)
		}
	}
	tests := []struct {
		recipe string
		comps  []Component
		want   string
	}{
		{"relayvsr", []Component{dit}, `needs a "vae" file`},
		{"seedvr2", []Component{dit, vae, {Slot: "upscaler", Path: "/m/x.pth"}}, `slot "upscaler" is not used by a video upscaler set`},
		{"wan", []Component{dit, vae}, "unknown video upscaler script"},
	}
	for _, tt := range tests {
		err := ValidateComponents(KindVUpscale, tt.recipe, tt.comps)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: err = %v, want %q", tt.recipe, err, tt.want)
		}
	}
}

func TestVUpscaleSlotsAndLabels(t *testing.T) {
	if got := KindSlots(KindVUpscale, "relayvsr").Slots(); strings.Join(got, ",") != "dit,vae" {
		t.Errorf("KindSlots = %v", got)
	}
	if got := SlotLabel("relayvsr", "vae"); got != "RelayVSR VAE (same scale)" {
		t.Errorf("SlotLabel = %q", got)
	}
	if got := SlotLabel("seedvr2", "dit"); got != "SeedVR2 diffusion model" {
		t.Errorf("SlotLabel = %q", got)
	}
}

func TestRelayVSRFileScale(t *testing.T) {
	for name, want := range map[string]int{
		"diffusion_models/relayvsr_x2_lightx2v_int8_convrot.safetensors": 2,
		"relayvsr_x4_int8_convrot.safetensors":                           4,
		"/models/relayvsr_flash_vae_x4_fp32.safetensors":                 4,
		"relayvsr-X2.safetensors":                                        2,
		"relayvsr_lightx2v.safetensors":                                  0,
		"my-upscaler.safetensors":                                        0,
	} {
		if got := RelayVSRFileScale(name); got != want {
			t.Errorf("RelayVSRFileScale(%q) = %d, want %d", name, got, want)
		}
	}
}

func TestVUpscaleCatalog(t *testing.T) {
	want := map[string]int{"relayvsr-x2": 2, "relayvsr-x4": 4, "seedvr2-3b": 2}
	for id, scale := range want {
		e, ok := LookupCatalog(id)
		if !ok {
			t.Fatalf("%s is not in the catalog", id)
		}
		if e.Kind != KindVUpscale {
			t.Errorf("%s: kind %s", id, e.Kind)
		}
		if err := ValidateComponents(e.Kind, e.Recipe, e.Components); err != nil {
			t.Errorf("%s: %v", id, err)
		}
		for _, c := range e.Components {
			if c.Bytes == 0 {
				t.Errorf("%s/%s has no size", id, c.Slot)
			}
		}
		if got := (ModelSet{CatalogID: id}).UpscaleScale(); got != scale {
			t.Errorf("%s: UpscaleScale = %d, want %d", id, got, scale)
		}
		if e.Recipe == "relayvsr" {
			if n := RelayVSRFileScale(e.Components[0].File); n != scale {
				t.Errorf("%s: DiT file is for ×%d", id, n)
			}
		}
	}
}

func vupscaleTestSet(t *testing.T, sameDir bool) ModelSet {
	t.Helper()
	d1, d2 := t.TempDir(), t.TempDir()
	if sameDir {
		d2 = d1
	}
	return ModelSet{ID: "v", Kind: KindVUpscale, Recipe: "relayvsr", Scale: 2, Complete: true, Components: []Component{
		{Slot: "vae", Path: filepath.Join(d2, "vae.safetensors")},
		{Slot: "dit", Path: filepath.Join(d1, "dit.safetensors")},
	}}
}

func TestLaunchSpecVUpscale(t *testing.T) {
	set := vupscaleTestSet(t, false)
	build := builder.BuildResult{ID: "sd-new-rocm", Profile: "rocm", BinaryPath: fakeSDServer(t, "--steps N\n  --dgl-dir DIR  Web UI script directory\n")}
	data := t.TempDir()
	inst := Instance{ID: "i", ModelSetID: "v", Port: 8081, ReservedMiB: 6 << 10, ExtraArgs: []string{"--threads", "4"}}
	cfg, err := LaunchSpec(inst, set, build, "", data, LaunchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--listen-ip", "0.0.0.0", "--listen-port", "8081",
		"--model-dir", filepath.Dir(set.Components[1].Path), "--model-dir", filepath.Dir(set.Components[0].Path),
		"--threads", "4"}
	if !slices.Equal(cfg.Args, want) {
		t.Errorf("args =\n%q\nwant\n%q", cfg.Args, want)
	}
	if cfg.HealthURL != "http://127.0.0.1:8081/v1/models" || cfg.WorkDir != InstanceDir(data, "i") {
		t.Errorf("health %s, work dir %s", cfg.HealthURL, cfg.WorkDir)
	}
	for _, d := range []string{"input", "output"} {
		if st, err := os.Stat(filepath.Join(cfg.WorkDir, d)); err != nil || !st.IsDir() {
			t.Errorf("%s folder: %v", d, err)
		}
	}
	if !slices.Contains(cfg.Env, "GGML_CUDA_REUSE_BUFFERS=0") {
		t.Errorf("env lacks GGML_CUDA_REUSE_BUFFERS=0: %v", cfg.Env)
	}

	// One folder for both files gives one --model-dir.
	cfg, err = LaunchSpec(inst, vupscaleTestSet(t, true), build, "", data, LaunchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(strings.Join(cfg.Args, " "), "--model-dir"); n != 1 {
		t.Errorf("%d --model-dir flags: %v", n, cfg.Args)
	}

	// A build without DGL is refused.
	old := builder.BuildResult{ID: "sd-1106-rocm", Profile: "rocm", BinaryPath: fakeSDServer(t, "--steps N\n")}
	if _, err := LaunchSpec(inst, set, old, "", data, LaunchOptions{}); err == nil || !strings.Contains(err.Error(), "need stable-diffusion.cpp master-1116 or newer (DGL); build sd-1106-rocm is older") {
		t.Errorf("old build: err = %v", err)
	}
}

func TestBuildHasDGL(t *testing.T) {
	for help, want := range map[string]bool{
		"--steps N\n  --dgl-dir DIR\n": true,
		"--steps N\n":                  false,
	} {
		build := builder.BuildResult{ID: "b", BinaryPath: fakeSDServer(t, help)}
		if got := BuildHasDGL(build); got != want {
			t.Errorf("help %q: %v, want %v", help, got, want)
		}
	}
	if !BuildHasDGL(builder.BuildResult{ID: "gone", BinaryPath: "/nonexistent/sd-server"}) {
		t.Error("an unreadable --help must count as DGL")
	}
}
