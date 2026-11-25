package api

// Measure for video upscaler instances: the admin chooses a test clip
// size (width, height, frames, before scaling) and a window size; toolchest
// writes a clip of that size, upscales it once alone on the GPU, and
// reserves the peak. Jobs larger than the measured clip are then refused.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/videofile"
)

// The Measure dialog's defaults: 5 s of Wan at 16 fps.
const (
	vupscaleMeasureWidth  = 832
	vupscaleMeasureHeight = 480
	vupscaleMeasureFrames = 81
	vupscaleMeasureFPS    = 16
)

// vupscaleMeasureSize reads the test clip's size and window from the
// Measure form, with the dialog's defaults for empty fields, and checks
// them as a job would be checked.
func vupscaleMeasureSize(r *http.Request, set generation.ModelSet) (generation.VideoSize, error) {
	num := func(name string, def int) (int, error) {
		v := strings.TrimSpace(r.FormValue(name))
		if v == "" {
			return def, nil
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return 0, fmt.Errorf("%s must be a whole number", name)
		}
		return n, nil
	}
	var size generation.VideoSize
	var err error
	if size.Width, err = num("width", vupscaleMeasureWidth); err != nil {
		return size, err
	}
	if size.Height, err = num("height", vupscaleMeasureHeight); err != nil {
		return size, err
	}
	if size.Frames, err = num("frames", vupscaleMeasureFrames); err != nil {
		return size, err
	}
	if size.Window, err = num("window_size", generation.DefaultWindowSize(set.Recipe)); err != nil {
		return size, err
	}
	for _, side := range []int{size.Width, size.Height} {
		if side < 64 || side > 2048 || side%16 != 0 {
			return size, errors.New("the clip width and height must be multiples of 16 from 64 to 2048")
		}
	}
	if size.Frames < 2 || size.Frames > 1024 {
		return size, errors.New("frames must be from 2 to 1024")
	}
	if err := generation.WindowSizeOK(set.Recipe, size.Window); err != nil {
		return size, err
	}
	info := videofile.Info{Width: size.Width, Height: size.Height, Frames: size.Frames}
	if e := checkVUpscaleClip(set.Recipe, set.UpscaleScale(), info, size.Window); e != nil {
		return size, errors.New(e.msg)
	}
	return size, nil
}

// measureClip is the test clip Measure upscales: a diagonal gradient that
// drifts, with a checker block that moves 4 pixels a frame, so the model
// has edges and motion to work on. Nothing ships with the program.
func measureClip(w, h, frames int) ([]byte, error) {
	var b bytes.Buffer
	err := videofile.WriteRGBAVI(&b, w, h, frames, vupscaleMeasureFPS, func(i int, rgb []byte) {
		bx, by := (i*4)%max(w-64, 1), h/3
		for y := range h {
			for x := range w {
				p := (y*w + x) * 3
				rgb[p] = byte((x + i*2) * 255 / max(w, 1))
				rgb[p+1] = byte(y * 255 / max(h, 1))
				rgb[p+2] = byte((x + y + i*3) / 4)
				if x >= bx && x < bx+64 && y >= by && y < by+64 && ((x-bx)/16+(y-by)/16)%2 == 0 {
					rgb[p], rgb[p+1], rgb[p+2] = 250, 250, 250
				}
			}
		}
	})
	return b.Bytes(), err
}

// runMeasureVUpscale upscales a test clip of the chosen size on the
// instance, which Measure has started without its reservation.
func (s *Server) runMeasureVUpscale(ctx context.Context, inst generation.Instance, set generation.ModelSet, size generation.VideoSize) error {
	inDir, outDir, err := generation.VUpscaleWorkDirs(s.cfg.DataDir, inst.ID)
	if err != nil {
		return err
	}
	clip, err := measureClip(size.Width, size.Height, size.Frames)
	if err != nil {
		return fmt.Errorf("making the test clip: %w", err)
	}
	in, out := filepath.Join(inDir, "measure.avi"), filepath.Join(outDir, "measure.webm")
	defer removeFiles(in, out)
	os.Remove(out) // sd-server does not overwrite a file another run left
	if err := os.WriteFile(in, clip, 0o644); err != nil {
		return err
	}
	src, err := generation.RenderVUpscale(set, generation.VUpscaleJob{
		InputPath: "input/measure.avi", OutputPath: "output/measure.webm", OutputFormat: "webm",
		Seed: 1, WindowSize: size.Window,
	})
	if err != nil {
		return err
	}
	if _, err := runDGLJob(ctx, inst.BaseURL(), src, nil); err != nil {
		if errors.Is(err, errDGLMemory) {
			return fmt.Errorf("the test upscale ran out of GPU memory at this size; try a smaller clip or window: %w", err)
		}
		return err
	}
	return nil
}

// vupscaleMeasureJSON is what a video upscaler instance's Measure dialog
// starts from: the last measured clip, or the defaults.
func vupscaleMeasureJSON(set generation.ModelSet, inst generation.Instance) string {
	v := map[string]any{
		"script": set.Recipe, "scale": set.UpscaleScale(),
		"width": vupscaleMeasureWidth, "height": vupscaleMeasureHeight, "frames": vupscaleMeasureFrames,
		"window": generation.DefaultWindowSize(set.Recipe),
	}
	if inst.MeasuredWindow > 0 {
		v["width"], v["height"], v["frames"], v["window"] = inst.MeasuredWidth, inst.MeasuredHeight, inst.MeasuredFrames, inst.MeasuredWindow
	}
	b, _ := json.Marshal(v)
	return string(b)
}
