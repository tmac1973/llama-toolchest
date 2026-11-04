//go:build linux

package monitor

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
)

// sysRoot is where sysfs is mounted. Tests point it at a fake tree.
var sysRoot = "/sys"

// pcieCache holds each card's link by PCI address. The topology and the
// negotiated link do not change while the machine runs, and the monitor
// asks every few seconds.
var pcieCache sync.Map // string → *PCIeLink

// bdfRE matches a PCI address as sysfs names its device directories,
// such as "0000:03:00.0".
var bdfRE = regexp.MustCompile(`^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$`)

// readPCIeLink returns the link between the card at bdf and the CPU, or
// nil when sysfs does not show it (no sysfs, a virtual machine that hides
// the topology, an address that is not there).
func readPCIeLink(bdf string) *PCIeLink {
	bdf = strings.ToLower(bdf)
	if v, ok := pcieCache.Load(bdf); ok {
		return v.(*PCIeLink)
	}
	link := scanPCIeLink(bdf)
	pcieCache.Store(bdf, link)
	return link
}

// scanPCIeLink follows the card's sysfs path up to the root port. Each
// device on the path, from the root port's child down to the card,
// reports the link to its parent. The card can only use what the
// slowest and narrowest of those links give it, and a link that also
// leads to other devices (a chipset, or a PCIe switch) is shared.
func scanPCIeLink(bdf string) *PCIeLink {
	resolved, err := filepath.EvalSymlinks(filepath.Join(sysRoot, "bus", "pci", "devices", bdf))
	if err != nil {
		return nil
	}
	// /sys/devices/pci0000:00/0000:00:01.2/0000:01:00.0/.../0000:03:00.0
	// gives the root port first and the card last.
	var path []string
	for _, part := range strings.Split(resolved, string(filepath.Separator)) {
		if bdfRE.MatchString(part) {
			path = append(path, part)
		}
	}
	if len(path) < 2 || path[len(path)-1] != bdf {
		return nil
	}
	dirs := make([]string, len(path))
	for i := range path {
		dirs[i] = filepath.Join(sysRoot, "bus", "pci", "devices", path[i])
	}

	link := &PCIeLink{}
	for i := 1; i < len(dirs); i++ {
		lanes := readSysInt(filepath.Join(dirs[i], "current_link_width"))
		gen := min(linkGen(dirs[i]), linkGen(dirs[i-1]))
		if lanes <= 0 || gen <= 0 {
			return nil
		}
		if link.Lanes == 0 || lanes < link.Lanes {
			link.Lanes = lanes
		}
		if link.Gen == 0 || gen < link.Gen {
			link.Gen = gen
		}
		// The card's own functions (its audio device, for one) share
		// its slot and are no other traffic.
		if i < len(dirs)-1 && hasOtherDevices(dirs[i], slotOf(bdf)) {
			link.Shared = true
		}
	}
	link.CardLanes = readSysInt(filepath.Join(dirs[len(dirs)-1], "max_link_width"))
	return link
}

// linkGen reads the highest PCIe generation a device supports from its
// max_link_speed. The current speed is no use: a card that is idle runs
// its link at the lowest speed to save power.
func linkGen(dir string) int {
	data, err := os.ReadFile(filepath.Join(dir, "max_link_speed"))
	if err != nil {
		return 0
	}
	gts, _, _ := strings.Cut(strings.TrimSpace(string(data)), " ")
	switch gts {
	case "2.5":
		return 1
	case "5.0", "5":
		return 2
	case "8.0", "8":
		return 3
	case "16.0", "16":
		return 4
	case "32.0", "32":
		return 5
	case "64.0", "64":
		return 6
	}
	return 0
}

// slotOf returns the bus and device part of an address ("0000:03:00"),
// which every function of one card shares.
func slotOf(bdf string) string {
	s, _, _ := strings.Cut(bdf, ".")
	return s
}

// hasOtherDevices reports whether the tree under dir holds a device that
// is not a bridge and not a function of the card in slot.
func hasOtherDevices(dir, slot string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if !bdfRE.MatchString(name) {
			continue
		}
		child := filepath.Join(dir, name)
		if isBridge(child) {
			if hasOtherDevices(child, slot) {
				return true
			}
			continue
		}
		if slotOf(name) != slot {
			return true
		}
	}
	return false
}

// isBridge reports whether the device at dir is a PCI-to-PCI bridge
// (class 0x0604).
func isBridge(dir string) bool {
	data, err := os.ReadFile(filepath.Join(dir, "class"))
	return err == nil && strings.HasPrefix(strings.TrimSpace(string(data)), "0x0604")
}

func readSysInt(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	return v
}
