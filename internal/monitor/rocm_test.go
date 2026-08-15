//go:build linux

package monitor

import (
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

// TestParseROCmGPUNamesSkipsCPU guards issue #68: rocminfo lists the host CPU
// as an agent, and the old name-blacklist let non-AMD CPUs (e.g. an Intel
// Xeon) leak through and get labelled as "GPU 0". Selecting by Device Type
// must drop the CPU agent regardless of its vendor.
func TestParseROCmGPUNamesSkipsCPU(t *testing.T) {
	// Abridged rocminfo output: an Intel Xeon CPU agent followed by an AMD GPU.
	out := `ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.1
*******
Agent 1
*******
  Name:                    Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Uuid:                    CPU-XX
  Marketing Name:          Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Device Type:             CPU
  Pool Info:
    Pool 1
      Segment:               GLOBAL; FLAGS: FINE GRAINED
*******
Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-XX
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
  Pool Info:
    Pool 1
      Segment:               GLOBAL; FLAGS: COARSE GRAINED
`
	got := parseROCmGPUNames(out)
	want := []string{"AMD Radeon RX 7900 XTX"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseROCmGPUNames = %v; want %v", got, want)
	}
}

// TestParseROCmGPUNamesMultiGPU verifies GPU agents are returned in order and
// the CPU between/around them is skipped.
func TestParseROCmGPUNamesMultiGPU(t *testing.T) {
	out := `*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
`
	got := parseROCmGPUNames(out)
	want := []string{"AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseROCmGPUNames = %v; want %v", got, want)
	}
}

// TestParseROCmGPUNamesFallbackToName covers a GPU agent with no marketing
// name — the gfx id stands in so the entry is never blank.
func TestParseROCmGPUNamesFallbackToName(t *testing.T) {
	out := `*******
Agent 1
*******
  Name:                    AMD Eng Sample
  Marketing Name:          AMD Eng Sample
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:
  Device Type:             GPU
`
	got := parseROCmGPUNames(out)
	want := []string{"gfx1151"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseROCmGPUNames = %v; want %v", got, want)
	}
}

// TestParseROCmGPUNamesNoGPU returns nil when only a CPU agent is present, so
// readGPUNameSysfs falls back to a generic label rather than the CPU name.
func TestParseROCmGPUNamesNoGPU(t *testing.T) {
	out := `*******
Agent 1
*******
  Name:                    Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Marketing Name:          Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Device Type:             CPU
`
	if got := parseROCmGPUNames(out); len(got) != 0 {
		t.Errorf("parseROCmGPUNames = %v; want empty", got)
	}
}

// TestKFDIndexByBDF guards the rocm-smi row mapping: rocm-smi orders
// its rows by PCI bus address while KFD topology (the order behind
// llama-server's ROCm<N> device names) can run the other way entirely
// — on a real quad-GPU board KFD listed c7, 83, 43, 03 while rocm-smi
// listed 03, 43, 83, c7. The PCI address is the only identity both
// share, so rows must resolve to KFD positions through it; matching by
// row position or by rocm-smi's "cardN" label paired one card's
// utilization with another card's VRAM (and produced duplicate GPU
// indices when a BMC display device shifted the DRM card numbering).
func TestKFDIndexByBDF(t *testing.T) {
	tmp := t.TempDir()
	// PCI device directories named by bus address, as sysfs does.
	bdfs := []string{"0000:c7:00.0", "0000:83:00.0", "0000:43:00.0", "0000:03:00.0"}
	var kfdDirs []string
	for i, bdf := range bdfs {
		target := filepath.Join(tmp, bdf)
		if err := os.Mkdir(target, 0o755); err != nil {
			t.Fatal(err)
		}
		link := filepath.Join(tmp, "renderD"+strconv.Itoa(128+i))
		if err := os.Symlink(target, link); err != nil {
			t.Fatal(err)
		}
		kfdDirs = append(kfdDirs, link)
	}

	m := kfdIndexByBDF(kfdDirs)
	if len(m) != 4 {
		t.Fatalf("kfdIndexByBDF returned %d entries; want 4", len(m))
	}
	// rocm-smi reports addresses uppercase; the map is keyed lowercase.
	for want, bdf := range bdfs {
		if got, ok := m[bdf]; !ok || got != want {
			t.Errorf("m[%q] = %d, %v; want %d, true", bdf, got, ok, want)
		}
	}
	if _, ok := m["0000:c9:00.0"]; ok {
		t.Error("BMC-style device not in KFD list must be absent from the map")
	}
}

// parseROCmGPUArchs must return the gfx target of each GPU agent, in
// the same order as parseROCmGPUNames, skipping CPU agents — it feeds
// the iGPU tagging that drives the model-config guard rails.
func TestParseROCmGPUArchs(t *testing.T) {
	out := `*******
Agent 1
*******
  Name:                    AMD Ryzen 7 9800X3D 8-Core Processor
  Marketing Name:          AMD Ryzen 7 9800X3D 8-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
  Device Type:             GPU
  Pool Info:
    Pool 1
      Name:                nested-pool-must-not-override
*******
Agent 3
*******
  Name:                    gfx1036
  Marketing Name:          AMD Ryzen 7 9800X3D 8-Core Processor
  Device Type:             GPU
`
	got := parseROCmGPUArchs(out)
	want := []string{"gfx1201", "gfx1036"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("parseROCmGPUArchs = %v; want %v", got, want)
	}
}
