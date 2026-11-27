package api

// /v1/images/upscale makes an image larger with an upscaler set's ESRGAN
// model. It answers at once (13 s for a 768x768 image on an R9700), so the
// route waits for the GPU like /v1/images/generations and has no job form.
//
// stable-diffusion.cpp master-1119 removed sd-server's /sdcpp/v1/upscale
// route (leejet/stable-diffusion.cpp#2404). On a build with DGL
// (generation.UpscaleUsesDGL) toolchest runs the upscale as a DGL script
// and gives the same answer the route gave; older builds still get the
// request forwarded to /sdcpp/v1/upscale.

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // input format
	"image/jpeg"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// upscaleMaxSide is the largest result side sd-server's upscale route
// allowed; the DGL path keeps it.
const upscaleMaxSide = 8192

// handleGenUpscale answers an upscale request. Its JSON fields are
// sd-server's: image (base64 or a data URL), upscaler, repeats, tile_size,
// output_format and output_compression, plus model to pick the instance.
func (s *Server) handleGenUpscale(w http.ResponseWriter, r *http.Request) {
	if isMultipartForm(r) {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request",
			"upscale takes JSON, with the image as base64 or a data URL")
		return
	}
	check := func(inst generation.Instance, _ string, body []byte) *requestError {
		if !s.upscaleUsesDGL(inst) {
			return nil // sd-server checks the request itself
		}
		set, err := s.gen.ModelSet(inst.ModelSetID)
		if err != nil {
			return &requestError{http.StatusInternalServerError, "server_error", err.Error()}
		}
		_, rerr := parseUpscaleRequest(body, set)
		return rerr
	}
	s.serveGenSyncWith(w, r, generation.KindUpscale, check, func(w http.ResponseWriter, r *http.Request, inst generation.Instance, body []byte) {
		if !s.upscaleUsesDGL(inst) {
			s.forwardToInstance(w, r, inst, "/sdcpp/v1/upscale", body)
			return
		}
		set, err := s.gen.ModelSet(inst.ModelSetID)
		if err != nil {
			openAIError(w, http.StatusInternalServerError, "server_error", "server_error", err.Error())
			return
		}
		req, rerr := parseUpscaleRequest(body, set)
		if rerr != nil {
			writeRequestError(w, rerr)
			return
		}
		// Not tied to the client, as forwardToInstance: the GPU stays
		// held until the upscale is really over.
		out, err := s.runUpscaleDGL(context.WithoutCancel(r.Context()), inst, set, req)
		if err != nil {
			openAIError(w, http.StatusBadGateway, "server_error", "backend_error", err.Error())
			return
		}
		respondJSONStatus(w, http.StatusOK, out)
	})
}

// upscaleUsesDGL reports whether inst runs on a build with DGL. When its
// build cannot be found, the instance cannot start either; the old route
// then gives that error.
func (s *Server) upscaleUsesDGL(inst generation.Instance) bool {
	if s.builder == nil {
		return false
	}
	b := s.resolveBackendBuild(string(generation.KindUpscale), inst.BuildID)
	return b != nil && generation.UpscaleUsesDGL(*b)
}

// upscaleRequest is a checked upscale request for the DGL path.
type upscaleRequest struct {
	image       image.Image
	repeats     int
	tileSize    int
	format      string // png or jpeg
	compression int    // JPEG quality, 0 to 100
}

// parseUpscaleRequest reads and checks an upscale request as sd-server's
// old route did: repeats is held to 1–4, tile_size is at least 32, and the
// result is at most 8192 pixels on each side. DGL saves only PNG, and
// toolchest makes JPEG from it; it cannot make WebP. The image may be PNG,
// JPEG or GIF.
func parseUpscaleRequest(body []byte, set generation.ModelSet) (upscaleRequest, *requestError) {
	var in struct {
		Image             string `json:"image"`
		Repeats           *int   `json:"repeats"`
		TileSize          *int   `json:"tile_size"`
		OutputFormat      string `json:"output_format"`
		OutputCompression *int   `json:"output_compression"`
	}
	if err := json.Unmarshal(body, &in); err != nil {
		return upscaleRequest{}, badRequest("bad_request", "the request is not valid JSON: %v", err)
	}
	req := upscaleRequest{repeats: 1, format: "png", compression: 100}
	if in.Repeats != nil {
		req.repeats = min(max(*in.Repeats, 1), 4)
	}
	if in.TileSize != nil {
		if *in.TileSize < 32 || *in.TileSize > 16384 {
			return upscaleRequest{}, badRequest("bad_value", "tile_size must be between 32 and 16384")
		}
		req.tileSize = *in.TileSize
	}
	switch f := strings.ToLower(in.OutputFormat); f {
	case "", "png":
	case "jpeg", "jpg":
		req.format = "jpeg"
	case "webp":
		return upscaleRequest{}, badRequest("bad_value", "output_format webp is not available with stable-diffusion.cpp master-1119 and newer; use png or jpeg")
	default:
		return upscaleRequest{}, badRequest("bad_value", "output_format must be png or jpeg")
	}
	if in.OutputCompression != nil {
		req.compression = min(max(*in.OutputCompression, 0), 100)
	}
	if in.Image == "" {
		return upscaleRequest{}, badRequest("bad_request", "image is required")
	}
	raw, err := base64.StdEncoding.DecodeString(stripDataURL(in.Image))
	if err != nil {
		return upscaleRequest{}, badRequest("bad_value", "image is not valid base64")
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return upscaleRequest{}, badRequest("bad_value", "the image could not be read; send PNG, JPEG or GIF")
	}
	req.image = img
	// A custom set may not say its scale; sd.cpp then is the only check.
	if scale := set.UpscaleScale(); scale > 0 {
		w, h := img.Bounds().Dx(), img.Bounds().Dy()
		for range req.repeats {
			w, h = w*scale, h*scale
		}
		if w > upscaleMaxSide || h > upscaleMaxSide {
			return upscaleRequest{}, badRequest("too_large", "the upscaled image would be %dx%d; it must not be larger than %dx%d. Send a smaller image or fewer repeats", w, h, upscaleMaxSide, upscaleMaxSide)
		}
	}
	return req, nil
}

// upscaleAnswer is the answer of sd-server's old /sdcpp/v1/upscale route.
type upscaleAnswer struct {
	GenerationInfo string         `json:"generation_info"`
	Upscaler       string         `json:"upscaler"`
	Scale          int            `json:"scale"`
	Repeats        int            `json:"repeats"`
	Width          int            `json:"width"`
	Height         int            `json:"height"`
	OutputFormat   string         `json:"output_format"`
	Images         []upscaleImage `json:"images"`
}

type upscaleImage struct {
	Index   int    `json:"index"`
	B64JSON string `json:"b64_json"`
}

// runUpscaleDGL upscales req.image on inst with the ESRGAN DGL script.
// The input and output files live in the instance folder only for the run.
func (s *Server) runUpscaleDGL(ctx context.Context, inst generation.Instance, set generation.ModelSet, req upscaleRequest) (upscaleAnswer, error) {
	inDir, outDir, err := generation.DGLWorkDirs(s.cfg.DataDir, inst.ID)
	if err != nil {
		return upscaleAnswer{}, err
	}
	var idb [8]byte
	_, _ = rand.Read(idb[:])
	name := "upscale-" + hex.EncodeToString(idb[:]) + ".png"
	in, out := filepath.Join(inDir, name), filepath.Join(outDir, name)
	defer removeFiles(in, out)
	// The upscale op takes only RGB (#737); the alpha goes back on after.
	rgb, alpha := splitAlpha(req.image)
	var buf bytes.Buffer
	if err := png.Encode(&buf, rgb); err != nil {
		return upscaleAnswer{}, fmt.Errorf("preparing the image: %w", err)
	}
	if err := os.WriteFile(in, buf.Bytes(), 0o644); err != nil {
		return upscaleAnswer{}, err
	}
	src, err := generation.RenderUpscale(set, generation.UpscaleJob{
		InputPath: "input/" + name, OutputPath: "output/" + name, TileSize: req.tileSize, Repeats: req.repeats,
	})
	if err != nil {
		return upscaleAnswer{}, err
	}
	if _, err := runDGLScript(ctx, inst.BaseURL(), src, nil); err != nil {
		var mem dglMemoryError
		if errors.As(err, &mem) {
			return upscaleAnswer{}, fmt.Errorf("sd.cpp ran out of GPU memory for this image; send a smaller image, fewer repeats or a smaller tile_size (sd.cpp: %s)", mem.sdcpp)
		}
		if errors.Is(err, errDGLBusy) {
			return upscaleAnswer{}, err
		}
		return upscaleAnswer{}, fmt.Errorf("upscaling: %w", err)
	}
	result, err := os.ReadFile(out)
	if err != nil {
		return upscaleAnswer{}, fmt.Errorf("sd-server's output is missing: %w", err)
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(result))
	if err != nil {
		return upscaleAnswer{}, fmt.Errorf("reading sd-server's output: %w", err)
	}
	switch {
	case req.format == "jpeg": // JPEG has no alpha
		img, err := png.Decode(bytes.NewReader(result))
		if err != nil {
			return upscaleAnswer{}, fmt.Errorf("reading sd-server's output: %w", err)
		}
		var j bytes.Buffer
		if err := jpeg.Encode(&j, img, &jpeg.Options{Quality: max(req.compression, 1)}); err != nil {
			return upscaleAnswer{}, fmt.Errorf("making the JPEG: %w", err)
		}
		result = j.Bytes()
	case alpha != nil:
		img, err := png.Decode(bytes.NewReader(result))
		if err != nil {
			return upscaleAnswer{}, fmt.Errorf("reading sd-server's output: %w", err)
		}
		var p bytes.Buffer
		if err := png.Encode(&p, withAlpha(img, alpha)); err != nil {
			return upscaleAnswer{}, fmt.Errorf("putting the alpha back: %w", err)
		}
		result = p.Bytes()
	}
	path := generation.UpscalerPath(set)
	return upscaleAnswer{
		Upscaler: strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Scale:    upscaleFactor(req.image.Bounds().Dx(), cfg.Width, req.repeats),
		Repeats:  req.repeats, Width: cfg.Width, Height: cfg.Height, OutputFormat: req.format,
		Images: []upscaleImage{{Index: 0, B64JSON: base64.StdEncoding.EncodeToString(result)}},
	}, nil
}

// upscaleFactor is the model's scale: the factor that, applied repeats
// times, makes width in into width out; 0 when none does.
func upscaleFactor(in, out, repeats int) int {
	for f := 1; f <= 16; f++ {
		w := in
		for range repeats {
			w *= f
		}
		if w == out {
			return f
		}
	}
	return 0
}
