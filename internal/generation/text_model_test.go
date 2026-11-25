package generation

import (
	"slices"
	"testing"
)

func TestTextModelToGPU(t *testing.T) {
	for _, tc := range []struct {
		in, want []string
	}{
		{[]string{"--steps", "20", "--backend", "te=cpu"}, []string{"--steps", "20"}},
		{[]string{"--backend", "te=cpu,vae=cpu", "--diffusion-fa"}, []string{"--backend", "vae=cpu", "--diffusion-fa"}},
		{[]string{"--backend=te=CPU"}, []string{}},
		{[]string{"--backend=vae=cpu,clip=cpu"}, []string{"--backend=vae=cpu"}},
		{[]string{"--clip-on-cpu", "--vae-tiling"}, []string{"--vae-tiling"}},
		// Other placements and a text model on another device stay.
		{[]string{"--backend", "vae=cpu"}, []string{"--backend", "vae=cpu"}},
		{[]string{"--backend", "te=rocm0"}, []string{"--backend", "te=rocm0"}},
		{[]string{"--backend", "cpu"}, []string{"--backend", "cpu"}},
		{nil, nil},
	} {
		got := TextModelToGPU(tc.in)
		if !slices.Equal(got, tc.want) {
			t.Errorf("TextModelToGPU(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if on := TextModelOnCPU(tc.in); on != !slices.Equal(got, tc.in) {
			t.Errorf("TextModelOnCPU(%q) = %v", tc.in, on)
		}
	}
}

func TestTextModelToGPUKeepsInput(t *testing.T) {
	in := []string{"--backend", "te=cpu,vae=cpu"}
	TextModelToGPU(in)
	if !slices.Equal(in, []string{"--backend", "te=cpu,vae=cpu"}) {
		t.Errorf("input changed: %q", in)
	}
}

// Every catalog sd-server set that runs its text model on the CPU offers
// the choice, and InstanceRunArgs drops only that placement.
func TestInstanceRunArgsCatalog(t *testing.T) {
	var offered int
	for _, e := range Catalog() {
		set := ModelSet{ID: e.ID, CatalogID: e.ID, Kind: e.Kind}
		cpu := InstanceRunArgs(set, Instance{})
		if !slices.Equal(cpu, e.LaunchArgs) {
			t.Errorf("%s: default run args %q, want the catalog's %q", e.ID, cpu, e.LaunchArgs)
		}
		gpu := InstanceRunArgs(set, Instance{TextModelGPU: true})
		if !set.TextModelOnCPU() {
			if !slices.Equal(gpu, e.LaunchArgs) {
				t.Errorf("%s: no te=cpu, but the GPU choice changed the args to %q", e.ID, gpu)
			}
			continue
		}
		offered++
		if TextModelOnCPU(gpu) || len(gpu) >= len(e.LaunchArgs) {
			t.Errorf("%s: GPU choice args %q still put the text model on the CPU", e.ID, gpu)
		}
	}
	if offered == 0 {
		t.Error("no catalog set offers the text model choice")
	}
}
