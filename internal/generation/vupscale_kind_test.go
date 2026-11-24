package generation

import (
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

func TestLaunchSpecVUpscaleNotYet(t *testing.T) {
	set := ModelSet{ID: "v", Kind: KindVUpscale, Recipe: "relayvsr", Complete: true}
	build := builder.BuildResult{ID: "b", Profile: "rocm", BinaryPath: "/b/sd-server"}
	if _, err := LaunchSpec(Instance{ID: "i", ModelSetID: "v", Port: 8081}, set, build, "", t.TempDir(), LaunchOptions{}); err == nil || !strings.Contains(err.Error(), "not supported yet") {
		t.Errorf("err = %v", err)
	}
}
