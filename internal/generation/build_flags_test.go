package generation

import (
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// Parts of sd-server --help before and after the Diffusion Graph
// Language rewrite (master-1015 and master-1116).
const (
	helpOld = "  --steps <int>                            number of sample steps (default: 20)\n" +
		"  --high-noise-steps <int>                 (high noise) number of sample steps (default: -1 = auto)\n" +
		"  --high-noise-cfg-scale <float>           (high noise) unconditional guidance scale: (default: 7.0)\n" +
		"  --high-noise-sampling-method             (high noise) sampling method\n" +
		"  --moe-boundary <float>                   timestep boundary for Wan2.2 MoE model.\n"
	helpNew = "  --steps <int>                            number of sample steps (default: 20)\n" +
		"  --high-noise-steps <int>                 high-noise steps; >= 0 adds to --steps, -1 splits --steps at --moe-boundary (default: -1)\n" +
		"  --moe-boundary <float>                   timestep boundary for Wan2.2 MoE model.\n"
)

func catalogArgs(t *testing.T, id string) []string {
	t.Helper()
	for _, e := range Catalog() {
		if e.ID == id {
			return e.LaunchArgs
		}
	}
	t.Fatalf("no catalog entry %s", id)
	return nil
}

func TestAdaptArgs(t *testing.T) {
	lightning := catalogArgs(t, "wan22-i2v-a14b-lightning")
	a14b := catalogArgs(t, "wan22-i2v-a14b")

	// An old build keeps every flag; so does a build that could not be
	// asked.
	for _, help := range []string{helpOld, ""} {
		for _, args := range [][]string{lightning, a14b} {
			if got, dropped := adaptArgs(args, help); !slices.Equal(got, args) || dropped != nil {
				t.Errorf("help %q: args changed to %q (dropped %v)", help[:min(len(help), 20)], got, dropped)
			}
		}
	}

	got, dropped := adaptArgs(lightning, helpNew)
	want := []string{"--steps", "4", "--high-noise-steps", "4", "--cfg-scale", "1",
		"--flow-shift", "5", "--sampling-method", "euler",
		"--diffusion-fa", "--temporal-tiling", "--backend", "te=cpu"}
	if !slices.Equal(got, want) {
		t.Errorf("Lightning on a new build =\n%q\nwant\n%q", got, want)
	}
	if want := []droppedFlag{{"--high-noise-cfg-scale", "1"}, {"--high-noise-sampling-method", "euler"}}; !slices.Equal(dropped, want) {
		t.Errorf("dropped %v, want %v", dropped, want)
	}

	got, _ = adaptArgs(a14b, helpNew)
	want = []string{"--steps", "20", "--cfg-scale", "3.5", "--flow-shift", "5", "--moe-boundary", "0.9",
		"--sampling-method", "euler", "--diffusion-fa", "--temporal-tiling", "--backend", "te=cpu"}
	if !slices.Equal(got, want) {
		t.Errorf("A14B on a new build =\n%q\nwant\n%q", got, want)
	}

	// The flag=value form, and a flag at the end with no value.
	got, dropped = adaptArgs([]string{"--high-noise-eta=0.5", "--steps", "8", "--high-noise-cfg-scale"}, helpNew)
	if !slices.Equal(got, []string{"--steps", "8"}) || len(dropped) != 2 || dropped[0].value != "0.5" {
		t.Errorf("got %q, dropped %v", got, dropped)
	}
}

// In the catalog sets, each high-noise flag either goes with each
// request on new builds (highNoiseRequestKeys) or has the same value as
// the flag the high-noise expert uses there, so leaving it out changes
// nothing.
func TestCatalogHighNoiseFlagsMatchMain(t *testing.T) {
	for _, e := range Catalog() {
		for flag, main := range highNoiseFlags {
			v, ok := argValue(e.LaunchArgs, flag)
			if !ok {
				continue
			}
			if _, sent := highNoiseRequestKeys[flag]; sent {
				continue
			}
			if mv, _ := argValue(e.LaunchArgs, main); mv != v {
				t.Errorf("%s: %s %s, but %s %q", e.ID, flag, v, main, mv)
			}
		}
	}
}

func TestHelpLists(t *testing.T) {
	cases := []struct {
		flag string
		want bool
	}{
		{"--high-noise-steps", true},
		{"--high-noise-cfg-scale", false},
		{"--steps", true},
		{"--moe", false},
	}
	for _, c := range cases {
		if got := helpLists(helpNew, c.flag); got != c.want {
			t.Errorf("helpLists(%s) = %v", c.flag, got)
		}
	}
	if !helpLists("-s, --steps", "--steps") || !helpLists("--steps=<int>", "--steps") {
		t.Error("missed a flag after a comma or before =")
	}
}

// fakeSDServer writes a script that prints help and exits 1, as some
// sd-server builds do.
func fakeSDServer(t *testing.T, help string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("needs a shell script")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "help.txt"), []byte(help), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(dir, "sd-server")
	script := "#!/bin/sh\ncat \"$(dirname \"$0\")/help.txt\"\nexit 1\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin
}

func TestLaunchSpecLeavesOutRemovedHighNoiseFlags(t *testing.T) {
	set := lightningSet("/m/loras/LOW.safetensors")
	set.LaunchArgs = catalogArgs(t, "wan22-i2v-a14b-lightning")
	inst := Instance{ID: "fast", Port: 8090}
	for _, c := range []struct {
		help string
		keep bool
	}{{helpOld, true}, {helpNew, false}, {"not sd-server\n", true}} {
		build := builder.BuildResult{ID: "sd-x-rocm", Profile: "rocm", BinaryPath: fakeSDServer(t, c.help)}
		cfg, err := LaunchSpec(inst, set, build, "", t.TempDir(), LaunchOptions{})
		if err != nil {
			t.Fatal(err)
		}
		_, has := argValue(cfg.Args, "--high-noise-cfg-scale")
		if has != c.keep {
			t.Errorf("help %q: --high-noise-cfg-scale kept = %v, want %v", c.help[:20], has, c.keep)
		}
		if v, _ := argValue(cfg.Args, "--high-noise-steps"); v != "4" {
			t.Errorf("help %q: --high-noise-steps = %q", c.help[:20], v)
		}
	}
}

// A binary rebuilt in place is asked again.
func TestBuildHelpCachePerBinary(t *testing.T) {
	bin := fakeSDServer(t, helpOld)
	build := builder.BuildResult{ID: "b", BinaryPath: bin}
	if !helpLists(buildHelp(build), "--high-noise-cfg-scale") {
		t.Fatal("old help not read")
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(bin), "help.txt"), []byte(helpNew), 0o644); err != nil {
		t.Fatal(err)
	}
	script := "#!/bin/sh\ncat \"$(dirname \"$0\")/help.txt\"\n# rebuilt\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	if helpLists(buildHelp(build), "--high-noise-cfg-scale") {
		t.Error("the rebuilt binary's help was not read again")
	}
}

func TestHighNoiseRequestParams(t *testing.T) {
	got := HighNoiseRequestParams(catalogArgs(t, "wan22-t2v-a14b"))
	want := map[string]any{"guidance": map[string]any{"txt_cfg": 4.0}, "sample_method": "euler"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("T2V A14B = %v, want %v", got, want)
	}
	// Every flag with a request field, in both forms; a value that is
	// not a number is left out; the last value wins.
	got = HighNoiseRequestParams([]string{"--high-noise-slg-scale=2.5", "--high-noise-skip-layer-start", "0.01",
		"--high-noise-skip-layer-end", "0.2", "--high-noise-eta", "x", "--high-noise-img-cfg-scale", "1.5",
		"--high-noise-guidance", "3.5", "--high-noise-cfg-scale", "2", "--high-noise-cfg-scale", "6"})
	want = map[string]any{"guidance": map[string]any{
		"txt_cfg": 6.0, "img_cfg": 1.5, "distilled_guidance": 3.5,
		"slg": map[string]any{"scale": 2.5, "layer_start": 0.01, "layer_end": 0.2},
	}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("all flags = %v, want %v", got, want)
	}
	if got := HighNoiseRequestParams([]string{"--steps", "20", "--high-noise-skip-layers", "[7,8]"}); got != nil {
		t.Errorf("no request fields = %v, want nil", got)
	}
}
