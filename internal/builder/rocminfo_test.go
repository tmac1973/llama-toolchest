package builder

import (
	"os"
	"reflect"
	"testing"
)

// The full rocminfo output of a desktop with an RX 9070 XT and the Ryzen
// iGPU, as rocminfo prints it (colour codes, trailing spaces, pool and ISA
// blocks and all). Only the GPU UUID is replaced. This is the shape every
// ROCm host produces, so the parser must give exactly the two GPUs, in
// agent order, with their own names and not a nested pool or ISA name.
func TestParseROCmGPUAgentsRealOutput(t *testing.T) {
	out, err := os.ReadFile("testdata/rocminfo-9070xt-igpu.txt")
	if err != nil {
		t.Fatal(err)
	}
	got := ParseROCmGPUAgents(string(out))
	want := []ROCmAgent{
		{Name: "gfx1201", MarketingName: "AMD Radeon RX 9070 XT"},
		{Name: "gfx1036", MarketingName: "AMD Ryzen 7 9800X3D 8-Core Processor"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ParseROCmGPUAgents = %+v\nwant %+v", got, want)
	}
}

// The edge cases one at a time. The agent order matters because it is the
// order of llama-server's ROCm<N> device names; the CPU agent must never
// count as a GPU (issue #68); and only the first "Name:" of a block is the
// agent's gfx target, which the build uses to pick GPU targets.
func TestParseROCmGPUAgents(t *testing.T) {
	cases := []struct {
		name string
		out  string
		want []ROCmAgent
	}{
		{name: "empty input", out: "", want: nil},
		{
			name: "only header lines, no agent blocks",
			out:  "ROCk module is loaded\n=====================\nHSA System Attributes\nRuntime Version:         1.18\n",
			want: nil,
		},
		{
			// A "Name:" or "Device Type:" before the first agent block
			// belongs to no agent and must not start one.
			name: "header lines before the first agent are ignored",
			out: `Name:                    not-an-agent
Device Type:             GPU
==========
HSA Agents
==========
*******
Agent 1
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
`,
			want: []ROCmAgent{{Name: "gfx1100", MarketingName: "AMD Radeon RX 7900 XTX"}},
		},
		{
			name: "CPU agent is skipped whatever its vendor",
			out: `*******
Agent 1
*******
  Name:                    Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Marketing Name:          Intel(R) Xeon(R) W-2225 CPU @ 4.10GHz
  Device Type:             CPU
`,
			want: nil,
		},
		{
			name: "several GPUs keep agent order around the CPU",
			out: `*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
*******
Agent 2
*******
  Name:                    AMD EPYC 9124 16-Core Processor
  Marketing Name:          AMD EPYC 9124 16-Core Processor
  Device Type:             CPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
*******
Agent 4
*******
  Name:                    gfx1030
  Marketing Name:          AMD Radeon RX 6800
  Device Type:             GPU
`,
			want: []ROCmAgent{
				{Name: "gfx1201", MarketingName: "AMD Radeon AI PRO R9700"},
				{Name: "gfx1100", MarketingName: "AMD Radeon RX 7900 XTX"},
				{Name: "gfx1030", MarketingName: "AMD Radeon RX 6800"},
			},
		},
		{
			name: "nested pool and ISA names do not replace the agent name",
			out: `*******
Agent 1
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon RX 9070 XT
  Vendor Name:             AMD
  Device Type:             GPU
  Pool Info:
    Pool 1
      Name:                    nested-pool
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1201
    ISA 2
      Name:                    amdgcn-amd-amdhsa--gfx12-generic
`,
			want: []ROCmAgent{{Name: "gfx1201", MarketingName: "AMD Radeon RX 9070 XT"}},
		},
		{
			name: "missing or blank marketing name stays empty",
			out: `*******
Agent 1
*******
  Name:                    gfx1151
  Marketing Name:
  Device Type:             GPU
*******
Agent 2
*******
  Name:                    gfx1100
  Device Type:             GPU
`,
			want: []ROCmAgent{{Name: "gfx1151"}, {Name: "gfx1100"}},
		},
		{
			// rocminfo pads lines with trailing spaces; Windows-style line
			// ends can come from a saved copy. Neither may leak into names.
			name: "trailing spaces and carriage returns are trimmed",
			out:  "*******\r\nAgent 1   \r\n*******\r\n  Name:   gfx1100   \r\n  Marketing Name:   AMD Radeon RX 7900 XTX   \r\n  Device Type:   GPU   \r\n",
			want: []ROCmAgent{{Name: "gfx1100", MarketingName: "AMD Radeon RX 7900 XTX"}},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := ParseROCmGPUAgents(c.out)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("ParseROCmGPUAgents = %+v\nwant %+v", got, c.want)
			}
		})
	}
}
