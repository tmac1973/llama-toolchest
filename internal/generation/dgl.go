package generation

import (
	"bytes"
	"embed"
	"fmt"
	"strings"
	"text/template"
)

// The DGL scripts video upscaler sets run, one per script (Recipe), and
// esrgan.dgl.tmpl, which image upscaler sets run on builds with DGL (see
// UpscaleUsesDGL). See the header of each file for its upstream source.
//
//go:embed dgl/*.dgl.tmpl
var dglFiles embed.FS

var dglTemplates = template.Must(template.ParseFS(dglFiles, "dgl/*.dgl.tmpl"))

// VUpscaleJob is what one video upscale job fills into its script.
type VUpscaleJob struct {
	// InputPath and OutputPath are relative to the instance folder,
	// sd-server's working folder: "input/<id>.<ext>" and
	// "output/<id>.webm" or ".avi".
	InputPath    string
	OutputPath   string
	OutputFormat string // "webm" or "avi"
	Seed         int64
	WindowSize   int
	// FPS is passed to video::load only when it is not 0. Giving a rate
	// makes video::load resample the clip to it, so it is set only for a
	// clip with no rate of its own (a still WebP), which video::load
	// cannot read without one.
	FPS int
}

// DefaultWindowSize is the window a script uses when a job gives none:
// RelayVSR's 16 frames, and one frame at a time for SeedVR2.
func DefaultWindowSize(script string) int {
	if script == "seedvr2" {
		return 1
	}
	return 16
}

// WindowSizeOK checks a job's window size for a script. RelayVSR's windows
// share one frame, so a window needs at least 2; SeedVR2 pads each window
// to 4n+1 frames, so only 1 and 4n+1 waste nothing.
func WindowSizeOK(script string, n int) error {
	switch script {
	case "relayvsr":
		if n < 2 || n > 32 {
			return fmt.Errorf("window_size must be between 2 and 32 for RelayVSR, not %d", n)
		}
	case "seedvr2":
		if n < 1 || n > 33 || (n-1)%4 != 0 {
			return fmt.Errorf("window_size must be 1, 5, 9, 13, 17, 21, 25, 29 or 33 for SeedVR2, not %d", n)
		}
	default:
		return fmt.Errorf("unknown video upscaler script %q", script)
	}
	return nil
}

// dglString quotes s as a DGL string. It refuses what the quoting cannot
// hold: a quote, a backslash, a newline or another control character.
// Model paths come from the set, so users choose them.
func dglString(s string) (string, error) {
	for _, r := range s {
		switch {
		case r == '"':
			return "", fmt.Errorf("path cannot be used in a DGL script: %q has a double quote", s)
		case r == '\\':
			return "", fmt.Errorf("path cannot be used in a DGL script: %q has a backslash", s)
		case r < 0x20 || r == 0x7f:
			return "", fmt.Errorf("path cannot be used in a DGL script: %q has a control character", s)
		}
	}
	return `"` + s + `"`, nil
}

// RenderVUpscale fills in a video upscaler set's script for one job.
func RenderVUpscale(set ModelSet, job VUpscaleJob) (string, error) {
	if set.Kind != KindVUpscale {
		return "", fmt.Errorf("set %s is not a video upscaler set", set.ID)
	}
	tmpl := dglTemplates.Lookup(set.Recipe + ".dgl.tmpl")
	if tmpl == nil {
		return "", fmt.Errorf("unknown video upscaler script %q", set.Recipe)
	}
	if err := WindowSizeOK(set.Recipe, job.WindowSize); err != nil {
		return "", err
	}
	if job.OutputFormat != "webm" && job.OutputFormat != "avi" {
		return "", fmt.Errorf("output format must be webm or avi, not %q", job.OutputFormat)
	}
	if job.FPS < 0 || job.FPS > 1000 {
		return "", fmt.Errorf("fps %d is out of range", job.FPS)
	}
	paths := map[string]string{}
	for _, c := range set.Components {
		paths[c.Slot] = c.Path
	}
	data := struct {
		DiT, VAE, Input, Output string
		Seed                    int64
		WindowSize, Scale, FPS  int
		OutputFormat            string
	}{Seed: job.Seed, WindowSize: job.WindowSize, Scale: set.UpscaleScale(), FPS: job.FPS, OutputFormat: job.OutputFormat}
	for _, f := range []struct {
		dst   *string
		value string
		name  string
	}{
		{&data.DiT, paths["dit"], "the diffusion model"},
		{&data.VAE, paths["vae"], "the VAE"},
		{&data.Input, job.InputPath, "the input"},
		{&data.Output, job.OutputPath, "the output"},
	} {
		if f.value == "" {
			return "", fmt.Errorf("the script needs %s's path", f.name)
		}
		q, err := dglString(f.value)
		if err != nil {
			return "", err
		}
		*f.dst = q
	}
	if set.Recipe == "seedvr2" && (data.Scale < 1 || data.Scale > 8) {
		return "", fmt.Errorf("set %s has no usable scale (%d)", set.ID, data.Scale)
	}
	var b bytes.Buffer
	if err := tmpl.Execute(&b, data); err != nil {
		return "", err
	}
	return strings.TrimLeft(b.String(), "\n"), nil
}

// UpscaleJob is what one image upscale fills into the ESRGAN script.
type UpscaleJob struct {
	// InputPath and OutputPath are relative to the instance folder,
	// sd-server's working folder: "input/<id>.png" and "output/<id>.png".
	// DGL saves only PNG.
	InputPath, OutputPath string
	// TileSize is the tile ESRGAN works on, 32 or more; 0 is sd.cpp's
	// default of 128.
	TileSize int
	// Repeats is how many times the image is upscaled, 1 to 4.
	Repeats int
}

// DefaultUpscaleTileSize is sd.cpp's default ESRGAN tile.
const DefaultUpscaleTileSize = 128

// RenderUpscale fills in the ESRGAN script for one image upscale with an
// upscaler set.
func RenderUpscale(set ModelSet, job UpscaleJob) (string, error) {
	if set.Kind != KindUpscale {
		return "", fmt.Errorf("set %s is not an upscaler set", set.ID)
	}
	if job.Repeats < 1 || job.Repeats > 4 {
		return "", fmt.Errorf("repeats must be between 1 and 4, not %d", job.Repeats)
	}
	if job.TileSize == 0 {
		job.TileSize = DefaultUpscaleTileSize
	}
	if job.TileSize < 32 || job.TileSize > 16384 {
		return "", fmt.Errorf("tile_size must be between 32 and 16384, not %d", job.TileSize)
	}
	if !strings.HasSuffix(job.OutputPath, ".png") {
		return "", fmt.Errorf("the output must be a .png file, not %q", job.OutputPath)
	}
	data := struct {
		Upscaler, Input, Output string
		TileSize, Repeats       int
	}{TileSize: job.TileSize, Repeats: job.Repeats}
	for _, f := range []struct {
		dst   *string
		value string
		name  string
	}{
		{&data.Upscaler, UpscalerPath(set), "the upscaler"},
		{&data.Input, job.InputPath, "the input"},
		{&data.Output, job.OutputPath, "the output"},
	} {
		if f.value == "" {
			return "", fmt.Errorf("the script needs %s's path", f.name)
		}
		q, err := dglString(f.value)
		if err != nil {
			return "", err
		}
		*f.dst = q
	}
	var b bytes.Buffer
	if err := dglTemplates.Lookup("esrgan.dgl.tmpl").Execute(&b, data); err != nil {
		return "", err
	}
	return strings.TrimLeft(b.String(), "\n"), nil
}

// UpscalerPath is the path of an upscaler set's ESRGAN file, or "".
func UpscalerPath(set ModelSet) string {
	for _, c := range set.Components {
		if c.Slot == UpscalerSlot {
			return c.Path
		}
	}
	return ""
}
