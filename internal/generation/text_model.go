package generation

import (
	"slices"
	"strings"
)

// Most sd-server sets run their text model on the CPU (--backend te=cpu):
// it saves 4-6 GiB of VRAM and costs a second or two on a plain image.
// Edits with several reference images pay much more, because the picture
// reader runs with the text model: a 3-image Qwen-Image-Edit 2511 edit
// spent 2 x 197 s conditioning on a Ryzen 9 5950X against 2 x 8 s on a
// Radeon AI PRO R9700, for about 5.5 GiB more VRAM (sd-master-1132,
// 2026-11-25). An instance can therefore run it on its GPU instead
// (Instance.TextModelGPU).

// InstanceRunArgs are the set's run arguments as inst runs them: without
// the text model's CPU placement when the instance puts it on the GPU.
func InstanceRunArgs(set ModelSet, inst Instance) []string {
	args := set.RunArgs()
	if inst.TextModelGPU {
		return TextModelToGPU(args)
	}
	return args
}

// TextModelOnCPU reports whether the set's flags put its text model on
// the CPU, so that an instance can choose to run it on the GPU instead.
func (m ModelSet) TextModelOnCPU() bool {
	return TextModelOnCPU(m.RunArgs())
}

// TextModelOnCPU reports whether args put the text model on the CPU:
// te=cpu in a --backend value, or the old --clip-on-cpu.
func TextModelOnCPU(args []string) bool {
	return !slices.Equal(TextModelToGPU(args), args)
}

// TextModelToGPU returns args without the text model's CPU placement, so
// sd-server runs it on the GPU with the rest of the model. It drops te=cpu
// from each --backend value (and the flag when nothing is left) and
// --clip-on-cpu. Other placements stay. args is not changed.
func TextModelToGPU(args []string) []string {
	out := make([]string, 0, len(args))
	changed := false
	for i := 0; i < len(args); i++ {
		a := args[i]
		switch {
		case a == "--clip-on-cpu":
			changed = true
			continue
		case a == "--backend" && i+1 < len(args):
			v := dropTextCPU(args[i+1])
			changed = changed || v != args[i+1]
			i++
			if v != "" {
				out = append(out, a, v)
			}
			continue
		case strings.HasPrefix(a, "--backend="):
			old := strings.TrimPrefix(a, "--backend=")
			v := dropTextCPU(old)
			changed = changed || v != old
			if v != "" {
				out = append(out, "--backend="+v)
			}
			continue
		}
		out = append(out, a)
	}
	if !changed {
		return args
	}
	return out
}

// dropTextCPU removes te=cpu (and clip=cpu, its older name) from a
// --backend value such as "te=cpu,vae=cpu".
func dropTextCPU(v string) string {
	parts := strings.Split(v, ",")
	kept := parts[:0:0]
	for _, p := range parts {
		k, dev, ok := strings.Cut(strings.TrimSpace(p), "=")
		if ok && (k == "te" || k == "clip") && strings.EqualFold(dev, "cpu") {
			continue
		}
		kept = append(kept, p)
	}
	return strings.Join(kept, ",")
}
