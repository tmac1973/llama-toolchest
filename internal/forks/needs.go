package forks

import (
	"fmt"
	"slices"
	"strings"
)

// HostInfo is what CheckNeeds compares a fork's needs with. Zero values
// mean unknown, and an unknown value gives no warning.
type HostInfo struct {
	RAMGiB int
	// AVX2Known reports whether the CPU flags were read; AVX2 is only
	// checked then.
	AVX2Known bool
	AVX2      bool
	// GPUArch is the chosen GPU's architecture ("gfx1100"), and Backend
	// its build profile backend ("rocm", "cuda").
	GPUArch string
	Backend string
	// FreeDiskGiB is the models folder's free space. Callers leave it
	// zero once the fork's model set is downloaded, since the space is
	// then already used.
	FreeDiskGiB int
}

// CheckNeeds returns one sentence for each need of f the host does not
// meet. An empty result means all known needs are met.
func CheckNeeds(f Fork, host HostInfo) []string {
	var out []string
	n := f.Needs
	if n.MinRAMGiB > 0 && host.RAMGiB > 0 && host.RAMGiB < n.MinRAMGiB {
		out = append(out, fmt.Sprintf("This host has %d GiB of RAM. %s needs %d GiB or more.", host.RAMGiB, f.Name, n.MinRAMGiB))
	}
	if n.AVX2 && host.AVX2Known && !host.AVX2 {
		out = append(out, fmt.Sprintf("This CPU does not have AVX2. %s needs it.", f.Name))
	}
	switch host.Backend {
	case "":
	case "rocm":
		if len(n.ROCmArchs) == 0 {
			out = append(out, fmt.Sprintf("%s does not support AMD GPUs.", f.Name))
		} else if host.GPUArch != "" && !slices.Contains(n.ROCmArchs, host.GPUArch) {
			out = append(out, fmt.Sprintf("This GPU is %s. %s is tuned for %s and may not run on it.",
				host.GPUArch, f.Name, strings.Join(n.ROCmArchs, ", ")))
		}
	case "cuda":
		if !n.CUDA {
			out = append(out, fmt.Sprintf("%s does not support NVIDIA GPUs.", f.Name))
		}
	default:
		if !slices.Contains(f.Backends, host.Backend) {
			out = append(out, fmt.Sprintf("%s does not support %s GPUs.", f.Name, host.Backend))
		}
	}
	if n.MinDiskGiB > 0 && host.FreeDiskGiB > 0 && host.FreeDiskGiB < n.MinDiskGiB {
		out = append(out, fmt.Sprintf("The models folder has %d GiB free. %s needs about %d GiB.", host.FreeDiskGiB, f.Name, n.MinDiskGiB))
	}
	return out
}
