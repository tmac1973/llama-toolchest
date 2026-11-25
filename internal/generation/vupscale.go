package generation

import (
	"os"
	"path/filepath"
	"slices"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// VUpscaleMinBuild is the first stable-diffusion.cpp release with DGL
// (sd-server's /sdcpp/v1/dgl/jobs) and the RelayVSR and SeedVR2 bindings.
// Upstream's example scripts for both have not changed since (checked up
// to master-1122).
const VUpscaleMinBuild = "master-1116"

// vupscaleMaxVRAMBuild is the first stable-diffusion.cpp master build
// (commit count) whose DGL engine runs video upscalers inside --max-vram
// (leejet/stable-diffusion.cpp#2419). Before it, a budget above the real
// peak still failed weight preparation: RelayVSR ×2 at 832x480 peaked at
// 9.3 GiB with no limit and failed inside 10.5 GiB (master-1116). On
// master-1132 the same job ran inside limits down to 5 GiB, a little
// slower, and SeedVR2 3B down to 6 GiB (RTX A4000, 2026-11-25).
const vupscaleMaxVRAMBuild = 1129

// VUpscaleTakesMaxVRAM reports whether a video upscaler instance on build
// runs inside --max-vram. A build with no recorded commit count (a branch
// or bare SHA from before commit counts were kept) runs without it.
func VUpscaleTakesMaxVRAM(build builder.BuildResult) bool {
	return build.CommitCount >= vupscaleMaxVRAMBuild
}

// UpscaleUsesDGL reports whether an image upscaler instance on build runs
// its upscales as a DGL script. stable-diffusion.cpp master-1119 removed
// sd-server's /sdcpp/v1/upscale route (5b80e1f, "use one context with
// on-demand model loading", leejet/stable-diffusion.cpp#2404); from then
// on ESRGAN runs only through DGL, which came in master-1116. Every build
// with DGL takes the script, so a branch or SHA build with no commit count
// works too.
func UpscaleUsesDGL(build builder.BuildResult) bool {
	return BuildHasDGL(build)
}

// DGLWorkDirs creates and returns the input and output folders of an
// instance that runs DGL scripts (a video upscaler, or an image upscaler
// on a build with DGL). sd-server runs in the instance folder, so a DGL
// script reads input/<job>.<ext> and writes output/<job>.<ext>: sd-server
// lets DGL write only under output/ of its working folder.
func DGLWorkDirs(dataDir, instID string) (in, out string, err error) {
	dir := InstanceDir(dataDir, instID)
	in, out = filepath.Join(dir, "input"), filepath.Join(dir, "output")
	for _, d := range []string{in, out} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return "", "", err
		}
	}
	return in, out, nil
}

// vupscaleModelDirs are the folders of a video upscaler set's files, in
// slot order (dit, then vae), each once.
func vupscaleModelDirs(set ModelSet) []string {
	return modelDirs(set, VUpscaleSlots)
}

// modelDirs are the folders of a set's files in the given slots, in slot
// order, each once: the --model-dir folders a DGL script loads them from.
// They are the folders of the files themselves, not of links to them.
func modelDirs(set ModelSet, slots []string) []string {
	var dirs []string
	for _, slot := range slots {
		for _, c := range set.Components {
			if c.Slot != slot || c.Path == "" {
				continue
			}
			if d := filepath.Dir(c.Path); !slices.Contains(dirs, d) {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}
