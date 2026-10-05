package builder

import "strings"

// ROCmAgent is one GPU agent from rocminfo output.
type ROCmAgent struct {
	Name          string // the agent name, which for a GPU is its gfx target (e.g. "gfx1100")
	MarketingName string // e.g. "AMD Radeon RX 7900 XTX"; can be empty
}

// ParseROCmGPUAgents returns the GPU agents in rocminfo output, in agent
// order. It is the one reader of this format: the build backend detection
// and the GPU monitor both use it, so they cannot disagree about which
// agents are GPUs or in what order they come.
//
// rocminfo lists every HSA agent, including the host CPU, under "Agent N"
// blocks, each with a "Device Type:" (CPU or GPU). Keying off that line is
// robust to the CPU vendor. An older approach skipped names starting with
// "AMD Ryzen" or "AMD EPYC", which let an Intel Xeon host show up as GPU 0
// (issue #68). Only the first "Name:" of a block is the agent's; later ones
// belong to nested pool and ISA entries.
func ParseROCmGPUAgents(out string) []ROCmAgent {
	type agent struct {
		ROCmAgent
		isGPU bool
	}
	var agents []agent
	cur := -1
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "Agent "):
			agents = append(agents, agent{})
			cur = len(agents) - 1
		case cur < 0:
			// Header lines before the first agent block.
			continue
		case strings.HasPrefix(line, "Marketing Name:"):
			agents[cur].MarketingName = strings.TrimSpace(strings.TrimPrefix(line, "Marketing Name:"))
		case strings.HasPrefix(line, "Name:"):
			if agents[cur].Name == "" {
				agents[cur].Name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			}
		case strings.HasPrefix(line, "Device Type:"):
			if strings.Contains(line, "GPU") {
				agents[cur].isGPU = true
			}
		}
	}

	var gpus []ROCmAgent
	for _, a := range agents {
		if a.isGPU {
			gpus = append(gpus, a.ROCmAgent)
		}
	}
	return gpus
}
