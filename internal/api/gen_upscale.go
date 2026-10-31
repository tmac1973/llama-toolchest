package api

// /v1/images/upscale makes an image larger with an upscaler set's ESRGAN
// model. sd-server's /sdcpp/v1/upscale answers at once (13 s for a 768x768
// image on an R9700), so the route waits for the GPU like
// /v1/images/generations and has no job form.

import (
	"net/http"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// handleGenUpscale forwards a JSON upscale request unchanged to an upscaler
// instance's /sdcpp/v1/upscale. Its fields are sd-server's: image (base64
// or a data URL), upscaler, repeats, tile_size, output_format and
// output_compression, plus model to pick the instance.
func (s *Server) handleGenUpscale(w http.ResponseWriter, r *http.Request) {
	if isMultipartForm(r) {
		openAIError(w, http.StatusBadRequest, "invalid_request_error", "bad_request",
			"upscale takes JSON, with the image as base64 or a data URL")
		return
	}
	s.serveGenSync(w, r, generation.KindUpscale, "/sdcpp/v1/upscale", nil)
}
