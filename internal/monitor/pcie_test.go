//go:build linux

package monitor

import (
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
)

// sysDev is one PCI device in a fake sysfs tree: its path under
// /sys/devices, and the files that describe it.
type sysDev struct {
	path      string // such as "pci0000:00/0000:00:01.2/0000:01:00.0"
	bridge    bool
	lanes     int // current_link_width
	maxLanes  int // max_link_width
	maxSpeed  string
	noLinkInf bool // a device without link files
}

// The board in compute2 (X570, Ryzen 9 3950X): GPU 0 behind the chipset,
// which also holds Wi-Fi and SATA, and GPU 1 on a root port of the CPU.
var compute2Tree = []sysDev{
	{path: "pci0000:00/0000:00:01.2", bridge: true, lanes: 4, maxLanes: 4, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0", bridge: true, lanes: 4, maxLanes: 8, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:01.0", bridge: true, lanes: 4, maxLanes: 4, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:01.0/0000:03:00.0", lanes: 4, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:01.0/0000:03:00.1", lanes: 4, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:03.0", bridge: true, lanes: 1, maxLanes: 4, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:03.0/0000:06:00.0", lanes: 1, maxLanes: 1, maxSpeed: "5.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:01.2/0000:01:00.0/0000:0b:00.0", noLinkInf: true},
	{path: "pci0000:00/0000:00:03.1", bridge: true, lanes: 8, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:03.1/0000:0e:00.0", lanes: 8, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	{path: "pci0000:00/0000:00:03.1/0000:0e:00.1", lanes: 8, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
}

// fakeSysfs builds tree under a temporary sysfs root, with the
// /sys/bus/pci/devices links the kernel makes, and empties the cache.
func fakeSysfs(t *testing.T, tree []sysDev) {
	t.Helper()
	root := t.TempDir()
	devices := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(devices, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(dir, name, value string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(value+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, d := range tree {
		dir := filepath.Join(root, "devices", d.path)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		class := "0x030000"
		if d.bridge {
			class = "0x060400"
		}
		write(dir, "class", class)
		if !d.noLinkInf {
			write(dir, "current_link_width", strconv.Itoa(d.lanes))
			write(dir, "max_link_width", strconv.Itoa(d.maxLanes))
			write(dir, "max_link_speed", d.maxSpeed)
		}
		if err := os.Symlink(dir, filepath.Join(devices, filepath.Base(d.path))); err != nil {
			t.Fatal(err)
		}
	}
	old := sysRoot
	sysRoot = root
	pcieCache = sync.Map{}
	t.Cleanup(func() {
		sysRoot = old
		pcieCache = sync.Map{}
	})
}

func TestPCIeLinkOnTheChipsetIsSharedAndNarrow(t *testing.T) {
	fakeSysfs(t, compute2Tree)
	got := readPCIeLink("0000:03:00.0")
	want := PCIeLink{Gen: 4, Lanes: 4, CardLanes: 16, Shared: true}
	if got == nil || *got != want {
		t.Fatalf("link = %+v, want %+v", got, want)
	}
}

// The card's own audio function is not other traffic.
func TestPCIeLinkOnACPURootPortIsNotShared(t *testing.T) {
	fakeSysfs(t, compute2Tree)
	got := readPCIeLink("0000:0e:00.0")
	want := PCIeLink{Gen: 4, Lanes: 8, CardLanes: 16}
	if got == nil || *got != want {
		t.Fatalf("link = %+v, want %+v", got, want)
	}
}

// A Radeon card has a PCIe switch of its own between the slot and the
// GPU. The GPU's link to that switch reports the full x16; the switch's
// link to the slot is what the card gets. The switch holds only the
// card, so nothing is shared.
func TestPCIeLinkThroughTheCardsOwnSwitch(t *testing.T) {
	fakeSysfs(t, []sysDev{
		{path: "pci0000:00/0000:00:01.1", bridge: true, lanes: 8, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
		{path: "pci0000:00/0000:00:01.1/0000:01:00.0", bridge: true, lanes: 8, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
		{path: "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0", bridge: true, lanes: 16, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
		{path: "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.0", lanes: 16, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
		{path: "pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0/0000:03:00.1", lanes: 16, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	})
	got := readPCIeLink("0000:03:00.0")
	want := PCIeLink{Gen: 4, Lanes: 8, CardLanes: 16}
	if got == nil || *got != want {
		t.Fatalf("link = %+v, want %+v", got, want)
	}
}

// A PCIe 4.0 card in a PCIe 3.0 slot runs at 3.0, whatever it supports.
func TestPCIeLinkTakesTheSlowerEnd(t *testing.T) {
	fakeSysfs(t, []sysDev{
		{path: "pci0000:00/0000:00:01.0", bridge: true, lanes: 16, maxLanes: 16, maxSpeed: "8.0 GT/s PCIe"},
		{path: "pci0000:00/0000:00:01.0/0000:01:00.0", lanes: 16, maxLanes: 16, maxSpeed: "16.0 GT/s PCIe"},
	})
	if got := readPCIeLink("0000:01:00.0").Label(); got != "PCIe 3.0 x16" {
		t.Errorf("label = %q, want PCIe 3.0 x16", got)
	}
}

// No sysfs entry, or one without link files, is no link rather than a
// zero one.
func TestPCIeLinkUnknown(t *testing.T) {
	fakeSysfs(t, compute2Tree)
	for _, bdf := range []string{"0000:99:00.0", "0000:0b:00.0"} {
		if got := readPCIeLink(bdf); got != nil {
			t.Errorf("readPCIeLink(%q) = %+v, want nil", bdf, got)
		}
	}
}

func TestPCIeLinkTitle(t *testing.T) {
	l := &PCIeLink{Gen: 4, Lanes: 4, CardLanes: 16, Shared: true}
	want := "PCIe 4.0 link to the CPU with 4 lanes. The card supports 16. " +
		"The link goes through the chipset or a PCIe switch, and other devices use it too. " +
		"A split model is fastest on GPUs with more lanes and no shared link."
	if got := l.Title(); got != want {
		t.Errorf("title =\n %q\nwant\n %q", got, want)
	}
}
