package generation

import (
	"fmt"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// mingNote is the part of every Ming-Image set's notes that both versions
// share.
const mingNote = "Makes PNG images with transparent backgrounds: the prompt must start with \"RGBA, 4-channel, transparent background.\", and the image must start from a fully transparent canvas at strength 0.9 (init_image and strength in the <sd_cpp_extra_args> block). Without both, images are opaque. Its 10 GB text model runs on the CPU, so it needs about 10.5 GiB of VRAM and fits a 16 GB card. Needs a stable-diffusion.cpp build from before master-1116, or master-1142 or newer."

// mingEntry returns a Ming-Image catalog set. The versions differ only in
// their DiT, text model and tokenizer download; the VAE, the recipe and the
// launch settings are the same. lead starts the notes.
func mingEntry(id, name string, dit, llm, tokenizer Component, lead string) CatalogEntry {
	return CatalogEntry{
		ID:     id,
		Name:   name,
		Kind:   KindSD,
		Recipe: "ming-image",
		Components: []Component{
			dit,
			llm,
			{Slot: "vae", Repo: "Comfy-Org/Ming-Image", File: "vae/ming_image_vae_bf16.safetensors", Bytes: 253816696},
			tokenizer,
		},
		// --backend te=cpu runs the 10 GB text model on the CPU, as
		// Haruspex does. Measured on a Radeon AI PRO R9700 on 2026-10-25
		// (0.1): peak 9.4 GiB instead of 18.6 GiB, and the same 20.7 s per
		// transparent 1024x1024 image.
		LaunchArgs: []string{"--steps", "12", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--backend", "te=cpu"},
		FlagNotes: map[string]string{
			"--backend te=cpu": "Runs the 10 GB text model on the CPU, as Haruspex does. On a Radeon AI PRO R9700 the peak went from 18.6 to 9.4 GiB, and each transparent image still took 20.7 s.",
		},
		Capabilities:  []string{"alpha"},
		License:       "MIT",
		Notes:         lead + mingNote,
		DefaultWidth:  2048,
		DefaultHeight: 2048,
	}
}

// stable-diffusion.cpp master-1116 (6cd0ef1, DGL) loads each model file in
// its own loader. The diffusion loader then no longer sees the text
// model's connector tensors, takes Ming-Image for Z-Image, and fails with
// "cap_pad_token not in model metadata". master-1142 (908965d,
// leejet/stable-diffusion.cpp#2441) tells them apart by Z-Image's pad
// tokens instead. Both commit counts are inclusive bounds of the broken
// range.
const (
	mingBrokenFirst = 1116
	mingBrokenLast  = 1141
)

// MingBuildProblem returns why build cannot run a set with the Ming-Image
// recipe, or "" when it can. A build with no recorded commit count (a
// branch or bare SHA) is let through: sd.cpp's own error shows if it is
// one of the broken ones.
func MingBuildProblem(recipe string, build builder.BuildResult) string {
	if recipe != "ming-image" || build.CommitCount < mingBrokenFirst || build.CommitCount > mingBrokenLast {
		return ""
	}
	return fmt.Sprintf("stable-diffusion.cpp builds master-1116 to master-1141 take Ming-Image for Z-Image and cannot load it; build %s is one of them. Build master-1142 or newer on the Builds page, or pick another build for this instance", build.ID)
}
