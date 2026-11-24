package generation

import (
	"os"
	"path/filepath"
	"slices"
)

// VUpscaleMinBuild is the first stable-diffusion.cpp release with DGL
// (sd-server's /sdcpp/v1/dgl/jobs) and the RelayVSR and SeedVR2 bindings.
// Upstream's example scripts for both have not changed since (checked up
// to master-1122).
const VUpscaleMinBuild = "master-1116"

// VUpscaleWorkDirs creates and returns a video upscaler instance's input
// and output folders. sd-server runs in the instance folder, so a DGL
// script reads input/<job>.<ext> and writes output/<job>.<ext>: sd-server
// lets DGL write only under output/ of its working folder.
func VUpscaleWorkDirs(dataDir, instID string) (in, out string, err error) {
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
	var dirs []string
	for _, slot := range VUpscaleSlots {
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
