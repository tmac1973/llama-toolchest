package monitor

import "testing"

// From "ioreg -r -d 1 -c IOAccelerator" on an M4 Mac mini with a model
// loaded, cut to the lines the parser reads and a few around them.
const ioregM4 = `+-o AGXAcceleratorG16G  <class AGXAcceleratorG16G, id 0x100000302, registered, matched, active, busy 0 (0 ms), retain 23>
    {
      "IOClass" = "AGXAcceleratorG16G"
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=6238306304,"Tiler Utilization %"=0,"recoveryCount"=0,"Renderer Utilization %"=0,"Device Utilization %"=37,"In use system memory"=12091392}
      "model" = "Apple M4"
      "gpu-core-count" = 10
    }
`

func TestParseIOAccelerator(t *testing.T) {
	s, ok := parseIOAccelerator(ioregM4)
	if !ok {
		t.Fatal("parse failed")
	}
	want := appleGPUStats{Model: "Apple M4", Cores: 10, UsedBytes: 6238306304, UtilPercent: 37}
	if s != want {
		t.Errorf("got %+v, want %+v", s, want)
	}
	if _, ok := parseIOAccelerator(""); ok {
		t.Error("empty output parsed as a GPU")
	}
}

func TestAppleWiredLimitMiB(t *testing.T) {
	for _, tt := range []struct{ mem, limit, want int }{
		{16 * 1024, 0, 10922},  // Metal: 10922 MiB
		{24 * 1024, 0, 18186},  // Metal: 18186 MiB
		{64 * 1024, 0, 48496},  // Metal: 49152 MiB; a little under is safe
		{24 * 1024, 20000, 20000},
		{24 * 1024, 99999, 24 * 1024}, // never more than the RAM
	} {
		if got := appleWiredLimitMiB(tt.mem, tt.limit); got != tt.want {
			t.Errorf("appleWiredLimitMiB(%d, %d) = %d, want %d", tt.mem, tt.limit, got, tt.want)
		}
	}
}

func TestAppleGPUInfo(t *testing.T) {
	s, _ := parseIOAccelerator(ioregM4)
	g := appleGPUInfo(s, 18186)
	if g.Name != "Apple M4 (10-core GPU)" || g.VRAMTotalMiB != 18186 || g.VRAMUsedMiB != 5949 || g.UtilPercent != 37 || g.IsIGPU {
		t.Errorf("got %+v", g)
	}
}
