package generation

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// The Details command is what an instance really runs, once the markers
// are replaced with the instance's own values. It comes from the same
// buildArgs that LaunchSpec uses, so this holds for every kind.
func TestDetailsCommandMatchesLaunchSpec(t *testing.T) {
	aceFiles := t.TempDir()
	ace := ModelSet{ID: "ace", Kind: KindAce, Complete: true}
	for _, slot := range AceSlots {
		p := filepath.Join(aceFiles, slot+".gguf")
		if err := os.WriteFile(p, []byte(slot), 0o644); err != nil {
			t.Fatal(err)
		}
		ace.Components = append(ace.Components, Component{Slot: slot, Path: p})
	}
	upFile := filepath.Join(t.TempDir(), "x4.safetensors")
	if err := os.WriteFile(upFile, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	upscaler := ModelSet{ID: "up", Kind: KindUpscale, Complete: true, Components: []Component{{Slot: UpscalerSlot, Path: upFile}}}

	build := builder.BuildResult{ID: "b", Profile: "rocm", BinaryPath: "/b/server"}
	for _, set := range []ModelSet{mingSet(), lightningSet("/m/loras/LOW.safetensors"), ace, upscaler} {
		data := t.TempDir()
		inst := Instance{ID: "i", Port: 8091, ReservedMiB: 10240}
		cfg, err := LaunchSpec(inst, set, build, "0.0.0.0", data, LaunchOptions{})
		if err != nil {
			t.Fatalf("%s: %v", set.ID, err)
		}
		d, err := DetailsForSet(set, t.TempDir())
		if err != nil {
			t.Fatalf("%s: %v", set.ID, err)
		}
		got := strings.NewReplacer(markerHost, "0.0.0.0", markerPort, "8091", markerVRAM, "10.0",
			markerInstance, InstanceDir(data, "i")).Replace(d.Command)
		want := d.Program + " " + strings.Join(cfg.Args, " ")
		if got != want {
			t.Errorf("%s: Details command with the markers replaced:\n%s\nLaunchSpec:\n%s", set.ID, got, want)
		}
	}
}

func TestDetailsForCatalogNotDownloaded(t *testing.T) {
	models := t.TempDir()
	e, _ := LookupCatalog("flux3-image-klein")
	d, err := DetailsForCatalog(e, models)
	if err != nil {
		t.Fatal(err)
	}
	if d.Program != "sd-server" || d.Recipe != "flux3-klein" || d.Custom || d.License == "" || d.Notes == "" {
		t.Errorf("details = %+v", d)
	}
	var slots []string
	for _, s := range d.Slots {
		slots = append(slots, s.Slot)
		if s.Present || !strings.HasPrefix(s.Path, models+"/") || s.Repo == "" || s.Label == s.Slot {
			t.Errorf("slot %+v: want a missing file under %s, with its repo and a label", s, models)
		}
	}
	if !slices.Equal(slots, []string{"diffusion_model", "llm", "vae"}) {
		t.Errorf("slots = %v, want the recipe's order", slots)
	}
	if !d.Slots[0].Required || d.Slots[2].Required || !strings.Contains(d.Slots[2].Label, "(one of: vae, tae)") {
		t.Errorf("required and one-of marks: %+v", d.Slots)
	}
	for _, f := range d.Flags {
		if f.Reason == "" {
			t.Errorf("flag %s %s has no reason", f.Flag, f.Value)
		}
	}
	if !strings.HasPrefix(d.Command, "sd-server --listen-ip <listen address> --listen-port <port> --diffusion-model "+models) ||
		!strings.HasSuffix(d.Command, "--max-vram <vram limit>") {
		t.Errorf("command = %s", d.Command)
	}
	var fixed []string
	for _, f := range d.FixedFlags {
		fixed = append(fixed, f.Flag)
		if f.Reason == "" {
			t.Errorf("fixed flag %s has no reason", f.Flag)
		}
	}
	if !slices.Equal(fixed, []string{"--listen-ip", "--listen-port", "--max-vram"}) {
		t.Errorf("fixed flags = %v", fixed)
	}

	// Once the file is there, it shows as present.
	if err := os.MkdirAll(filepath.Dir(d.Slots[2].Path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(d.Slots[2].Path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	d, _ = DetailsForCatalog(e, models)
	if !d.Slots[2].Present || d.Slots[0].Present {
		t.Errorf("present = %v, %v", d.Slots[0].Present, d.Slots[2].Present)
	}
}

// A custom set with LoRAs: the LoRA folder is a flag toolchest adds, and
// the LoRA files are slots with labels.
func TestDetailsForCustomSetWithLoras(t *testing.T) {
	d, err := DetailsForSet(lightningSet("/m/loras/LOW.safetensors"), t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if !d.Custom || d.License != "" {
		t.Errorf("custom = %v, licence %q", d.Custom, d.License)
	}
	if !slices.Contains(d.FixedFlags, FlagDetail{Flag: "--lora-model-dir", Value: "/m/loras", Reason: FlagReasons["--lora-model-dir"]}) {
		t.Errorf("fixed flags = %+v, want --lora-model-dir /m/loras", d.FixedFlags)
	}
	labels := map[string]string{}
	for _, s := range d.Slots {
		labels[s.Slot] = s.Label
	}
	if labels["lora"] != "LoRA" || labels["high_noise_lora"] != "High-noise LoRA" || labels["diffusion_model"] != "Low-noise expert" {
		t.Errorf("labels = %v", labels)
	}
	if len(d.Flags) != 1 || d.Flags[0] != (FlagDetail{Flag: "--steps", Value: "4", Reason: FlagReasons["--steps"]}) {
		t.Errorf("flags = %+v, want the set's --steps 4 with the shared reason", d.Flags)
	}
}

func TestDetailsForCatalogMusic(t *testing.T) {
	e, _ := LookupCatalog("ace-step-xl-sft")
	d, err := DetailsForCatalog(e, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if d.Program != "ace-server" || len(d.Slots) != 4 || len(d.Flags) != 0 {
		t.Errorf("details = %+v", d)
	}
	if !slices.ContainsFunc(d.FixedFlags, func(f FlagDetail) bool { return f.Flag == "--max-batch" && f.Value == "1" }) ||
		!slices.ContainsFunc(d.FixedFlags, func(f FlagDetail) bool { return f.Flag == "--models" && f.Value == aceModelsFolder }) {
		t.Errorf("fixed flags = %+v", d.FixedFlags)
	}
	if d.Command != "ace-server --host <listen address> --port <port> --models <instance folder>/models --max-batch 1" {
		t.Errorf("command = %s", d.Command)
	}
}

// A catalog set that is not fully downloaded has no path for its missing
// files: Details shows where they will be saved, and the stored set keeps
// its empty paths.
func TestDetailsForSetFillsMissingPaths(t *testing.T) {
	e, _ := LookupCatalog("z-image-turbo")
	set := ModelSet{ID: e.ID, Name: e.Name, Kind: e.Kind, Recipe: e.Recipe, CatalogID: e.ID, Components: slices.Clone(e.Components)}
	models := t.TempDir()
	d, err := DetailsForSet(set, models)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range d.Slots {
		if !strings.HasPrefix(s.Path, models+"/") {
			t.Errorf("slot %s path %q, want one under the models folder", s.Slot, s.Path)
		}
	}
	if set.Components[0].Path != "" {
		t.Error("DetailsForSet changed the caller's set")
	}
	// A catalog set runs with the catalog's current flags, so Details
	// shows them, not the ones stored at download.
	if !slices.ContainsFunc(d.Flags, func(f FlagDetail) bool { return f.Flag == "--backend" && f.Value == "te=cpu" }) {
		t.Errorf("flags = %+v, want the catalog's --backend te=cpu", d.Flags)
	}
}

func TestShellWord(t *testing.T) {
	for in, want := range map[string]string{
		"/m/a.gguf":                "/m/a.gguf",
		"/my models/a.gguf":        "'/my models/a.gguf'",
		"low quality, ugly":        "'low quality, ugly'",
		"it's":                     `'it'\''s'`,
		"<listen address>":         "<listen address>",
		"<instance folder>/models": "<instance folder>/models",
		"te=cpu":                   "te=cpu",
	} {
		if got := shellWord(in); got != want {
			t.Errorf("shellWord(%q) = %s, want %s", in, got, want)
		}
	}
}
