//go:build fieldtest

package api

import (
	"context"
	"os"
	"strconv"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
)

// TestDGLField runs a RelayVSR or SeedVR2 script on a real, running
// video upscaler instance, for field tests only:
//
//	VUPSCALE_BASE_URL=http://127.0.0.1:8081 VUPSCALE_SCRIPT=relayvsr VUPSCALE_SCALE=2 \
//	VUPSCALE_DIT=/data/models/gen-models/.../dit.safetensors VUPSCALE_VAE=/data/.../vae.safetensors \
//	VUPSCALE_INPUT=input/clip.webm go test -tags fieldtest -run TestDGLField -v ./internal/api/
//
// VUPSCALE_INPUT is relative to the instance folder (sd-server's working
// folder); the result goes to output/field.webm there.
func TestDGLField(t *testing.T) {
	base := os.Getenv("VUPSCALE_BASE_URL")
	if base == "" {
		t.Skip("set VUPSCALE_BASE_URL and the other VUPSCALE_ variables")
	}
	script := os.Getenv("VUPSCALE_SCRIPT")
	scale, _ := strconv.Atoi(os.Getenv("VUPSCALE_SCALE"))
	set := generation.ModelSet{ID: "field", Kind: generation.KindVUpscale, Recipe: script, Scale: scale, Components: []generation.Component{
		{Slot: "dit", Path: os.Getenv("VUPSCALE_DIT")},
		{Slot: "vae", Path: os.Getenv("VUPSCALE_VAE")},
	}}
	src, err := generation.RenderVUpscale(set, generation.VUpscaleJob{
		InputPath: os.Getenv("VUPSCALE_INPUT"), OutputPath: "output/field.webm", OutputFormat: "webm",
		Seed: 1, WindowSize: generation.DefaultWindowSize(script),
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := runDGLJob(context.Background(), base, src, func(p dglProgress) { t.Logf("progress %+v", p) })
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("result %+v", res)
}
