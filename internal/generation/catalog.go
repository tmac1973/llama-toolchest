package generation

// CatalogEntry is a known-good model set that downloads in one click.
type CatalogEntry struct {
	ID   string      `json:"id"`
	Name string      `json:"name"`
	Kind BackendKind `json:"kind"`
	// Recipe is the sd-server recipe ID; empty for ace.
	Recipe       string      `json:"recipe,omitempty"`
	Components   []Component `json:"components"`
	LaunchArgs   []string    `json:"launch_args,omitempty"`
	Capabilities []string    `json:"capabilities,omitempty"`
	License      string      `json:"license"`
	// Notes is shown as a tooltip on the entry.
	Notes string `json:"notes,omitempty"`
	// DefaultWidth and DefaultHeight are the image or video size the
	// measured test run uses; zero for ace.
	DefaultWidth  int `json:"default_width,omitempty"`
	DefaultHeight int `json:"default_height,omitempty"`
	// DefaultFrames and DefaultFPS are a video set's test length and frame
	// rate; MaxFrames, MaxWidth and MaxHeight its ceiling for requests.
	DefaultFrames int `json:"default_frames,omitempty"`
	DefaultFPS    int `json:"default_fps,omitempty"`
	MaxFrames     int `json:"max_frames,omitempty"`
	MaxWidth      int `json:"max_width,omitempty"`
	MaxHeight     int `json:"max_height,omitempty"`
	// VideoSizes are the sizes Measure offers, the test size first and
	// none past the ceiling. FullMeasureTime is a sentence for the Measure
	// dialog on how long measuring the largest size takes, e.g.
	// "Measuring the largest size took about 21 minutes on a Radeon AI PRO
	// R9700."
	VideoSizes      []VideoSize `json:"video_sizes,omitempty"`
	FullMeasureTime string      `json:"full_measure_time,omitempty"`
	// SizeMultiple is what sd-server rounds a video's width and height up
	// to: the VAE's scale factor times the diffusion model's down factor
	// (sd.cpp master-1015, request.cpp). Zero when not known.
	SizeMultiple int `json:"size_multiple,omitempty"`
}

// TotalBytes is the download size of the whole set.
func (e CatalogEntry) TotalBytes() int64 {
	var n int64
	for _, c := range e.Components {
		n += c.Bytes
	}
	return n
}

// aceRepo holds every ACE-Step catalog file.
const aceRepo = "Serveurperso/ACE-Step-1.5-GGUF"

// aceComponents returns an ACE-Step set's four files. The language model,
// text embedding and VAE are shared by every variant; only the DiT differs.
func aceComponents(ditFile string, ditBytes int64) []Component {
	return []Component{
		{Slot: "lm", Repo: aceRepo, File: "acestep-5Hz-lm-4B-Q8_0.gguf", Bytes: 4457323648},
		{Slot: "dit", Repo: aceRepo, File: ditFile, Bytes: ditBytes},
		{Slot: "embedding", Repo: aceRepo, File: "Qwen3-Embedding-0.6B-BF16.gguf", Bytes: 1196890592},
		{Slot: "vae", Repo: aceRepo, File: "vae-BF16.gguf", Bytes: 337420928},
	}
}

// Files shared by more than one set. A shared file downloads once.
var (
	// fluxAE is the FLUX.1 VAE that Z-Image and Chroma use. The copy in
	// black-forest-labs/FLUX.1-schnell is gated; this one is not.
	fluxAE = Component{Slot: "vae", Repo: "Comfy-Org/z_image_turbo", File: "split_files/vae/ae.safetensors", Bytes: 335304388}
	// chromaT5 is the text encoder both Chroma sets use.
	chromaT5 = Component{Slot: "t5xxl", Repo: "city96/t5-v1_1-xxl-encoder-gguf", File: "t5-v1_1-xxl-encoder-Q8_0.gguf", Bytes: 5061584064}
	// wan21VAE is the Wan2.1 VAE, used by the Wan2.2 A14B pairs.
	wan21VAE = Component{Slot: "vae", Repo: "Comfy-Org/Wan_2.1_ComfyUI_repackaged", File: "split_files/vae/wan_2.1_vae.safetensors", Bytes: 253815318}
	// a14bLow and a14bHigh are the Wan2.2 I2V A14B experts, used by the
	// 20-step set and the Lightning set.
	a14bLow  = Component{Slot: "diffusion_model", Repo: "QuantStack/Wan2.2-I2V-A14B-GGUF", File: "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf", Bytes: 9651728896}
	a14bHigh = Component{Slot: "high_noise_diffusion_model", Repo: "QuantStack/Wan2.2-I2V-A14B-GGUF", File: "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf", Bytes: 9651728896}
	// umt5Q8 is the text encoder of the Wan video sets.
	umt5Q8 = Component{Slot: "t5xxl", Repo: "city96/umt5-xxl-encoder-gguf", File: "umt5-xxl-encoder-Q8_0.gguf", Bytes: 6043068256}
	// flux2SmallDecoder is a FLUX.2 VAE, used by FLUX.2 klein and FLUX.3
	// Image Klein. The FLUX.2-dev VAE is gated; this one is ungated and
	// Apache-licensed.
	flux2SmallDecoder = Component{Slot: "vae", Repo: "black-forest-labs/FLUX.2-small-decoder", File: "full_encoder_small_decoder.safetensors", Bytes: 249519092}
	// qwen25VL is Qwen2.5-VL 7B's language part: Qwen-Image Edit 2511's
	// text model, and HunyuanVideo 1.5's main one.
	qwen25VL = Component{Slot: "llm", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", File: "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf", Bytes: 4683072512}
	// hunyuanByT5 (the glyph text model, for writing in the video) and
	// hunyuanVAE are shared by every HunyuanVideo 1.5 set.
	hunyuanByT5 = Component{Slot: "t5xxl", Repo: hunyuanRepo, File: "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors", Bytes: 438643184}
	hunyuanVAE  = Component{Slot: "vae", Repo: hunyuanRepo, File: "split_files/vae/hunyuanvideo15_vae_fp16.safetensors", Bytes: 2521292758}
)

// hunyuanRepo is Comfy-Org's repackaged HunyuanVideo 1.5 files (once
// named Comfy-Org/HunyuanVideo_1.5_repackaged, which now redirects here).
const hunyuanRepo = "Comfy-Org/Hunyuan-Video-1.5"

// hunyuanLicense is the HunyuanVideo 1.5 sets' licence.
const hunyuanLicense = "Tencent Hunyuan Community License (does not apply in the EU, the UK or South Korea)"

// hunyuanEntry is one HunyuanVideo 1.5 set (Tencent, November 2025, 8.3B):
// its own video model with the shared text models and VAE. hd says the
// model is a 720p one (1280×720); otherwise 480p (848×480). steps, cfg and
// shift are Tencent's settings for that version (the model card's table).
// Every version makes up to 121 frames, 5 s at 24 fps; frames go in steps
// of 4 plus 1 (the recipe's video_constraints). measureTime is the Measure
// dialog's sentence on how long measuring Full takes.
func hunyuanEntry(id, name, recipe string, model Component, hd bool, steps, cfg, shift, notes, measureTime string) CatalogEntry {
	w, h := 848, 480
	if hd {
		w, h = 1280, 720
	}
	caps := []string{CapText}
	if recipe == "hunyuan-video-15-i2v" {
		caps = []string{CapStartImage}
	}
	return CatalogEntry{
		ID:         id,
		Name:       name,
		Kind:       KindVideo,
		Recipe:     recipe,
		Components: []Component{model, qwen25VL, hunyuanByT5, hunyuanVAE},
		LaunchArgs: []string{"--steps", steps, "--cfg-scale", cfg, "--flow-shift", shift, "--sampling-method", "euler",
			"--diffusion-fa", "--backend", "te=cpu"},
		Capabilities:  caps,
		License:       hunyuanLicense,
		Notes:         notes + " Shares its text models and VAE with the other HunyuanVideo 1.5 sets, and Qwen2.5-VL with Qwen-Image-Edit 2511. Licence: Tencent Hunyuan Community License, which does not apply in the EU, the UK or South Korea; terms at https://huggingface.co/tencent/HunyuanVideo-1.5.",
		DefaultWidth:  w,
		DefaultHeight: h,
		DefaultFrames: 33,
		DefaultFPS:    24,
		MaxFrames:     121,
		MaxWidth:      w,
		MaxHeight:     h,
		VideoSizes: []VideoSize{
			{Label: "Short", Frames: 33, Width: w, Height: h},
			{Label: "Medium", Frames: 65, Width: w, Height: h},
			{Label: "Full", Frames: 121, Width: w, Height: h},
		},
		FullMeasureTime: measureTime,
		// The HunyuanVideo 1.5 VAE scales by 16 (the recipe's size_multiple).
		SizeMultiple: 16,
	}
}

// hunyuan480Fit is what the 480p HunyuanVideo 1.5 sets need on a Radeon AI
// PRO R9700 (32 GB), measured 2026-10-29 with the text model on the CPU:
// every size peaked at 24-26 GiB, as sd.cpp keeps more of the model loaded
// when the card has room.
const hunyuan480Fit = "On a Radeon AI PRO R9700 (32 GB), every size up to 5 s fits, reserving about 27.5 GiB."

// hunyuan720Fit and hunyuan720Measure say how the 720p HunyuanVideo 1.5
// sets fared on a Radeon AI PRO R9700 (32 GB), measured 2026-10-29 with
// the text model on the CPU. The untiled decode at 1280x720 needs 31.4 GB,
// so sd.cpp falls back to decoding in smaller pieces (225 s for 33 frames,
// no seams). Short reserves 13-14.5 GiB; at 65 frames that decode ran out
// of memory, and at 121 frames stable-diffusion.cpp master-1015 crashed
// during sampling with "CONCAT failed".
const (
	hunyuan720Fit     = "On a Radeon AI PRO R9700 (32 GB), only the Short size (33 frames) works, reserving about 14.5 GiB: the decode is done in smaller pieces, as the whole 720p decode needs more than the card has, and longer videos ran out of memory or crashed in stable-diffusion.cpp."
	hunyuan720Measure = "On a Radeon AI PRO R9700, measuring Short took about 5 minutes. Medium ran out of memory in the decode, and Full crashed in stable-diffusion.cpp."
)

// hunyuanGGUF is a HunyuanVideo 1.5 video model from jayn7's GGUF repos,
// at Q8_0 (9.0 GB each), as TI2V 5B uses.
func hunyuanGGUF(repo, file string) Component {
	return Component{Slot: "diffusion_model", Repo: "jayn7/HunyuanVideo-1.5_" + repo + "-GGUF", File: file, Bytes: 8995313888}
}

// Every sd-server set but SDXL runs its text encoder on the CPU
// (--backend te=cpu). On a Radeon AI PRO R9700 with a Ryzen 9 5950X
// (2026-10-28/29) Measure's reservations went down by 4-5.5 GiB on each
// image set at no real cost (the Qwen encoders were faster on the CPU, as
// nothing was copied to the GPU), 6 GiB on TI2V 5B, 3 GiB on LTX-2.3 at
// Full (which then fitted) and about 1 GiB on Wan2.2 A14B, for 13-16 s a
// video. Qwen-Image Edit's picture reader goes with its text model (sd.cpp
// places only diffusion, te and vae), adding about 17 s to an edit. SDXL's
// two CLIP encoders are small and were not tried.
//
// Catalog returns the built-in model sets. File sizes were read from the
// Hugging Face tree API on 2026-10-24 (2026-10-25 for Z-Image Turbo,
// FLUX.3 Image Klein, Chroma1-HD, Chroma1-Flash and Qwen-Image-Edit 2511;
// 2026-10-26 for Wan2.2 TI2V 5B);
// launch settings for Qwen-Image 2.1 Turbo and ACE-Step XL SFT are the
// ones measured on asmodean on 2026-10-24.
func Catalog() []CatalogEntry {
	return []CatalogEntry{
		{
			ID:     "ming-image",
			Name:   "Ming-Image 0.1 Design",
			Kind:   KindSD,
			Recipe: "ming-image",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "voltaire321/Ming-Image-0.1-Design-GGUF", File: "ming_image_0.1_design-Q8_0.gguf", Bytes: 6541979808},
				{Slot: "llm", Repo: "voltaire321/Ming-Image-0.1-Design-GGUF", File: "ming_ling_mini_2.0-Q4_K.gguf", Bytes: 10518620096},
				{Slot: "vae", Repo: "Comfy-Org/Ming-Image", File: "vae/ming_image_vae_bf16.safetensors", Bytes: 253816696},
				{Slot: "tokenizer", Repo: "inclusionAI/Ming-Image-0.1-Design", File: "mllm/tokenizer.json", Bytes: 12210709},
			},
			// --backend te=cpu runs the 10 GB text model on the CPU, as
			// Haruspex does. Measured on a Radeon AI PRO R9700 on
			// 2026-10-25: peak 9.4 GiB instead of 18.6 GiB, and the same
			// 20.7 s per transparent 1024x1024 image.
			LaunchArgs:   []string{"--steps", "12", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--backend", "te=cpu"},
			Capabilities: []string{"alpha"},
			License:      "MIT",
			Notes: "Used by Haruspex. Can make PNG images with transparent backgrounds: the prompt must start with " +
				"\"RGBA, 4-channel, transparent background.\" and the image must start from a fully transparent canvas at strength 0.9 " +
				"(sd-server's <sd_cpp_extra_args> block, with init_image and strength). Without both, images come back opaque. " +
				"Its 10 GB text model runs on the CPU, so it needs about 10.5 GiB of GPU memory and fits on a 16 GB card.",
			DefaultWidth:  2048,
			DefaultHeight: 2048,
		},
		{
			ID:     "qwen-image-2.1-turbo",
			Name:   "Qwen-Image 2.1 Turbo (Q4)",
			Kind:   KindSD,
			Recipe: "qwen-image-21-turbo",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "unsloth/Qwen-Image-2.1-Turbo-GGUF", File: "qwen-image-2.1-turbo-Q4_K_M.gguf", Bytes: 4199565120},
				{Slot: "llm", Repo: "Qwen/Qwen3-VL-8B-Instruct-GGUF", File: "Qwen3VL-8B-Instruct-Q4_K_M.gguf", Bytes: 5027784800},
				{Slot: "vae", Repo: "Comfy-Org/Qwen-Image-2.1", File: "vae/qwen_image_2.1_vae_bf16.safetensors", Bytes: 675509688},
			},
			LaunchArgs: []string{"--sigmas", "qwen-image-2.1-turbo", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--backend", "te=cpu"},
			// No "edit": Qwen-Image edits need the llm_vision weights,
			// which this set leaves out.
			License:       "Qwen-Image 2.1 licence",
			Notes:         "8 steps; about 36 s per 1024×1024 image on a Radeon AI PRO R9700, reserving about 6.5 GiB of VRAM with its text model on the CPU (11 GiB with it on the GPU). Licence terms: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "flux2-klein-4b",
			Name:   "FLUX.2 klein 4B",
			Kind:   KindSD,
			Recipe: "flux2-klein-4b",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "leejet/FLUX.2-klein-4B-GGUF", File: "flux-2-klein-4b-Q8_0.gguf", Bytes: 4300629440},
				{Slot: "llm", Repo: "unsloth/Qwen3-4B-GGUF", File: "Qwen3-4B-Q8_0.gguf", Bytes: 4280405792},
				flux2SmallDecoder,
			},
			LaunchArgs:    []string{"--steps", "4", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--backend", "te=cpu"},
			Capabilities:  []string{"edit"},
			License:       "Apache-2.0",
			Notes:         "4 steps. Also edits images.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "z-image-turbo",
			Name:   "Z-Image Turbo",
			Kind:   KindSD,
			Recipe: "z-image-turbo",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "leejet/Z-Image-Turbo-GGUF", File: "z_image_turbo-Q8_0.gguf", Bytes: 6577440704},
				{Slot: "llm", Repo: "unsloth/Qwen3-4B-Instruct-2507-GGUF", File: "Qwen3-4B-Instruct-2507-Q8_0.gguf", Bytes: 4280405600},
				fluxAE,
			},
			LaunchArgs:    []string{"--steps", "9", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--backend", "te=cpu"},
			License:       "Apache-2.0",
			Notes:         "9 steps; about 19 s per 1024×1024 image on a Radeon AI PRO R9700, reserving about 8.5 GiB of VRAM with its text model on the CPU. Shares its VAE with Chroma1-HD.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "flux3-image-klein",
			Name:   "FLUX.3 Image Klein",
			Kind:   KindSD,
			Recipe: "flux3-klein",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "leejet/flux-3-image-klein-GGUF", File: "flux3-image-klein-Q8_0.gguf", Bytes: 6151716032},
				{Slot: "llm", Repo: "Qwen/Qwen3-VL-4B-Instruct-GGUF", File: "Qwen3VL-4B-Instruct-Q8_0.gguf", Bytes: 4280406144},
				flux2SmallDecoder,
			},
			LaunchArgs:    []string{"--steps", "16", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--backend", "te=cpu"},
			Capabilities:  []string{"edit"},
			License:       "FLUX Kommunity License (non-commercial)",
			Notes:         "16 steps; about 21 s per 1024×1024 image (56 s for an edit) on a Radeon AI PRO R9700, reserving about 9.5 GiB of VRAM with its text model on the CPU. Also edits images, with up to 10 reference images. Shares its VAE with FLUX.2 klein 4B. Non-commercial licence: only people and companies with less than US$5 million yearly revenue may use the images commercially. Licence terms: https://huggingface.co/black-forest-labs/flux-3-image-klein",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "chroma1-hd",
			Name:   "Chroma1-HD",
			Kind:   KindSD,
			Recipe: "chroma",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "silveroxides/Chroma-GGUF", File: "Chroma1-HD/Chroma1-HD-Q8_0.gguf", Bytes: 9735409824},
				chromaT5,
				fluxAE,
			},
			// Without a negative prompt, Chroma's images came out covered
			// in grainy noise on 2026-10-25; 20 steps still left some, 40
			// did not. --negative-prompt sets the default for requests that
			// give none.
			LaunchArgs: []string{"--steps", "40", "--cfg-scale", "4", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling",
				"--negative-prompt", "low quality, ugly, unfinished, out of focus, deformed, disfigured, blurry, smudged, restricted palette, flat colors", "--backend", "te=cpu"},
			License:       "Apache-2.0",
			Notes:         "40 steps; about 3.5 minutes per 1024×1024 image on a Radeon AI PRO R9700, reserving about 12 GiB of VRAM with its text model on the CPU (about 10 s more per image, for its negative prompt). Based on FLUX.1-schnell. Starts with a default negative prompt; a request can give its own. Shares its text model and VAE with Chroma1-Flash, and its VAE with Z-Image Turbo.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "chroma1-flash",
			Name:   "Chroma1-Flash",
			Kind:   KindSD,
			Recipe: "chroma",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "silveroxides/Chroma1-Flash-GGUF", File: "Chroma1-HD-Flash-Q8_0.gguf", Bytes: 9735409824},
				chromaT5,
				fluxAE,
			},
			// Flash has guidance trained in, so it runs at CFG 1: one pass
			// per step, and negative prompts do nothing. On 2026-10-25, 8
			// Euler steps garbled text in the image; 16 did not.
			LaunchArgs:    []string{"--steps", "16", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--backend", "te=cpu"},
			License:       "Apache-2.0",
			Notes:         "Faster Chroma1-HD: 16 steps; about 46 s per 1024×1024 image on a Radeon AI PRO R9700 (Chroma1-HD takes about 3.5 minutes), reserving about 12 GiB of VRAM with its text model on the CPU. Ignores negative prompts. Shares its text model and VAE with Chroma1-HD.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "qwen-image-edit-2511",
			Name:   "Qwen-Image-Edit 2511 (Q4)",
			Kind:   KindSD,
			Recipe: "qwen-image-edit-2511",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "unsloth/Qwen-Image-Edit-2511-GGUF", File: "qwen-image-edit-2511-Q4_K_M.gguf", Bytes: 13244758624},
				qwen25VL,
				{Slot: "llm_vision", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", File: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf", Bytes: 1354162912},
				{Slot: "vae", Repo: "Comfy-Org/Qwen-Image_ComfyUI", File: "split_files/vae/qwen_image_vae.safetensors", Bytes: 253806246},
			},
			// Without qwen_image_zero_cond_t=true, sd.cpp's docs say edit
			// quality drops a lot.
			LaunchArgs:    []string{"--steps", "20", "--cfg-scale", "4", "--flow-shift", "3", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--model-args", "qwen_image_zero_cond_t=true", "--backend", "te=cpu"},
			Capabilities:  []string{"edit"},
			License:       "Apache-2.0",
			Notes:         "Edits images from one or more reference images; 20 steps. About 6 minutes per 1024×1024 edit on a Radeon AI PRO R9700, reserving about 16.5 GiB of VRAM with its text model (and its picture reader) on the CPU, which adds about 17 s to an edit.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "sdxl-base",
			Name:   "Stable Diffusion XL base 1.0",
			Kind:   KindSD,
			Recipe: "sdxl-checkpoint",
			Components: []Component{
				{Slot: "model", Repo: "stabilityai/stable-diffusion-xl-base-1.0", File: "sd_xl_base_1.0.safetensors", Bytes: 6938078334},
			},
			LaunchArgs:    []string{"--steps", "20", "--cfg-scale", "5", "--vae-tiling"},
			License:       "openrail++",
			Notes:         "A single checkpoint file; 20 steps.",
			DefaultWidth:  1024,
			DefaultHeight: 1024,
		},
		{
			ID:     "wan22-ti2v-5b",
			Name:   "Wan2.2 TI2V 5B",
			Kind:   KindVideo,
			Recipe: "wan22-ti2v-5b",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "QuantStack/Wan2.2-TI2V-5B-GGUF", File: "Wan2.2-TI2V-5B-Q8_0.gguf", Bytes: 5400179040},
				umt5Q8,
				{Slot: "vae", Repo: "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", File: "split_files/vae/wan2.2_vae.safetensors", Bytes: 1409400960},
			},
			// No --vae-tiling: with it, sd.cpp (master-1000 and master-1006,
			// ROCm) crashed decoding the Wan2.2 VAE on a Radeon AI PRO R9700
			// on 2026-10-26. Without it the decode works but needs more
			// memory: the whole card peaked at about 25 GiB for the test
			// video.
			LaunchArgs:   []string{"--steps", "20", "--cfg-scale", "5", "--flow-shift", "5", "--sampling-method", "euler", "--diffusion-fa", "--backend", "te=cpu"},
			Capabilities: []string{CapText, CapStartImage},
			License:      "Apache-2.0",
			Notes: "Text-to-video and image-to-video in one small model; 20 steps. On a Radeon AI PRO R9700 (32 GB), about 70 s for the 33-frame 832×480 Short size " +
				"(reservation 20 GiB, with its text model on the CPU). Medium and Full need more memory than a 32 GB card can give: the decode alone needs more than the card has left, " +
				"so Measure refuses them there; custom sizes between Short and Medium can be tried. On a 16 GB RTX A4000, Short works inside 14.7 GiB (about 75 s).",
			// The test video is short and mid-sized so Measure finishes in
			// minutes; the ceiling is the model's native size, 121 frames
			// being 5 s at 24 fps.
			DefaultWidth:  832,
			DefaultHeight: 480,
			DefaultFrames: 33,
			DefaultFPS:    24,
			MaxFrames:     121,
			MaxWidth:      1280,
			MaxHeight:     704,
			VideoSizes: []VideoSize{
				{Label: "Short", Frames: 33, Width: 832, Height: 480},
				{Label: "Medium", Frames: 81, Width: 1280, Height: 704},
				{Label: "Full", Frames: 121, Width: 1280, Height: 704},
			},
			// Measured 2026-10-29 with the one-step Measure: Short in 43 s;
			// Medium stopped after 43 s, needing more than 29.4 GiB in the
			// decode against a 28.2 GiB room.
			FullMeasureTime: "On a Radeon AI PRO R9700, measuring Short took under a minute; Medium needed more memory than that 32 GB card could give, and Measure stopped it within a minute.",
			// The Wan2.2 VAE scales by 16 and the model patches by 2.
			SizeMultiple: 32,
		},
		{
			ID:     "wan22-i2v-a14b",
			Name:   "Wan2.2 I2V A14B (Q4)",
			Kind:   KindVideo,
			Recipe: "wan22-i2v",
			Components: []Component{
				a14bLow,
				a14bHigh,
				umt5Q8,
				wan21VAE,
			},
			// The wan22-i2v recipe's defaults. sd-server has no flag for the
			// recipe's high_noise_flow_shift; --flow-shift covers both
			// experts. --moe-boundary is the recipe's 0.9, not sd-server's
			// default 0.875.
			LaunchArgs: []string{"--steps", "20", "--cfg-scale", "3.5", "--flow-shift", "5", "--moe-boundary", "0.9",
				"--high-noise-cfg-scale", "3.5", "--high-noise-sampling-method", "euler",
				"--sampling-method", "euler", "--diffusion-fa", "--temporal-tiling", "--backend", "te=cpu"},
			// --temporal-tiling, not --vae-tiling: with --vae-tiling the
			// Wan2.1 VAE hung encoding the start picture on a Radeon AI PRO
			// R9700 (sd.cpp master-1015, 2026-10-26). Without any tiling the
			// decode took the whole card to 32.3 GB at 33 frames of 832×480,
			// over the 28.5 GiB the instance could have; --temporal-tiling
			// tiles only the decode and kept the card at 27.9 GB.
			Capabilities: []string{CapStartImage, CapEndImage},
			License:      "Apache-2.0",
			Notes: "Image-to-video and start-and-end videos; needs a start image. Two 14B experts (high and low noise); 20 steps. " +
				"Up to 5 seconds at 1280×720. Shares its text model with Wan2.2 TI2V 5B. " +
				"Slow: a 2-second video at 832×480 takes about 27 minutes on a Radeon AI PRO R9700, against about 70 seconds for TI2V 5B. " +
				"For faster videos, use Wan2.2 I2V A14B Lightning.",
			// 81 frames is 5 s at 16 fps.
			DefaultWidth:  832,
			DefaultHeight: 480,
			DefaultFrames: 33,
			DefaultFPS:    16,
			MaxFrames:     81,
			MaxWidth:      1280,
			MaxHeight:     720,
			VideoSizes: []VideoSize{
				{Label: "Short", Frames: 33, Width: 832, Height: 480},
				{Label: "Medium", Frames: 49, Width: 1280, Height: 720},
				{Label: "Full", Frames: 81, Width: 1280, Height: 720},
			},
			// Short measured in 123 s on 2026-10-29 (one step on each
			// expert); a real Short video with start and end images took
			// 1,650 s. Medium and Full not measured yet.
			FullMeasureTime: "On a Radeon AI PRO R9700, measuring Short took about 2 minutes; larger sizes have more video to decode and take longer.",
			// The Wan2.1 VAE scales by 8 and the model patches by 2.
			SizeMultiple: 16,
		},
		{
			ID:     "wan22-i2v-a14b-lightning",
			Name:   "Wan2.2 I2V A14B Lightning (Q4, 4 steps)",
			Kind:   KindVideo,
			Recipe: "wan22-i2v",
			// The Lightning 4-step LoRAs, one for each expert. These are
			// Kijai's fp16 copies of lightx2v's Wan2.2-Lightning I2V
			// Seko-V1 weights (Apache-2.0): lightx2v's own files (F32, with
			// I64 alpha values) made sd-server abort as they loaded
			// (sd.cpp master-1015, 2026-10-26). Both are in one folder,
			// which becomes --lora-model-dir.
			Components: []Component{
				a14bLow,
				a14bHigh,
				umt5Q8,
				wan21VAE,
				{Slot: "high_noise_lora", Repo: "Kijai/WanVideo_comfy", File: "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", Bytes: 613561776},
				{Slot: "lora", Repo: "Kijai/WanVideo_comfy", File: "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", Bytes: 613561776},
			},
			// 4 steps on each expert at CFG 1. The high-noise expert takes
			// the first 4 steps, so --moe-boundary is not used. Lightning's
			// own workflow uses 2 + 2, which was faster (110 s) but softer,
			// smearing the face in the last frames on a Radeon AI PRO R9700;
			// 4 + 4 (200 s) looked about as sharp as the 20-step set.
			// --temporal-tiling as for Wan2.2 I2V A14B.
			LaunchArgs: []string{"--steps", "4", "--high-noise-steps", "4", "--cfg-scale", "1", "--high-noise-cfg-scale", "1",
				"--flow-shift", "5", "--sampling-method", "euler", "--high-noise-sampling-method", "euler",
				"--diffusion-fa", "--temporal-tiling", "--backend", "te=cpu"},
			Capabilities: []string{CapStartImage, CapEndImage},
			License:      "Apache-2.0",
			Notes: "Wan2.2 I2V A14B with the Lightning 4-step LoRAs: image-to-video and start-and-end videos; needs a start image. " +
				"4 steps on each expert: about 3 minutes 20 seconds for a 2-second video at 832×480 on a Radeon AI PRO R9700, " +
				"against about 27 for the 20-step set, with a little less detail. Up to 5 seconds at 1280×720. " +
				"Shares its experts, text model and VAE with Wan2.2 I2V A14B.",
			DefaultWidth:  832,
			DefaultHeight: 480,
			DefaultFrames: 33,
			DefaultFPS:    16,
			MaxFrames:     81,
			MaxWidth:      1280,
			MaxHeight:     720,
			VideoSizes: []VideoSize{
				{Label: "Short", Frames: 33, Width: 832, Height: 480},
				{Label: "Medium", Frames: 49, Width: 1280, Height: 720},
				{Label: "Full", Frames: 81, Width: 1280, Height: 720},
			},
			// Medium was measured on the R9700 (2026-10-27): 25 minutes and
			// a 30.5 GiB peak, more than the card could give, so it was
			// refused. Full was not measured: it has more video still. With
			// the one-step Measure (2026-10-29), Short took 83 s; a real
			// Short video with start and end images took 200 s.
			FullMeasureTime: "On a Radeon AI PRO R9700, measuring Short took under 1.5 minutes; " +
				"Medium needed more memory than that 32 GB card could give.",
			// The Wan2.1 VAE scales by 8 and the model patches by 2.
			SizeMultiple: 16,
		},
		{
			ID:     "ltx23-distilled",
			Name:   "LTX-2.3 distilled 1.1 (Q4)",
			Kind:   KindVideo,
			Recipe: "ltx23-distilled",
			Components: []Component{
				{Slot: "diffusion_model", Repo: "unsloth/LTX-2.3-GGUF", File: "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf", Bytes: 14194441248},
				{Slot: "llm", Repo: "unsloth/gemma-3-12b-it-GGUF", File: "gemma-3-12b-it-Q4_K_M.gguf", Bytes: 7300778336},
				{Slot: "embeddings_connectors", Repo: "unsloth/LTX-2.3-GGUF", File: "text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors", Bytes: 2312144712},
				{Slot: "vae", Repo: "unsloth/LTX-2.3-GGUF", File: "vae/ltx-2.3-22b-dev_video_vae.safetensors", Bytes: 1452256522},
				{Slot: "audio_vae", Repo: "unsloth/LTX-2.3-GGUF", File: "vae/ltx-2.3-22b-dev_audio_vae.safetensors", Bytes: 364853140},
			},
			// The ltx23-distilled recipe's defaults. LTX sizes are multiples
			// of 32: sd-server rounds other sizes up (1280×720 becomes
			// 1280×736), which the user chose to allow. --temporal-tiling
			// and not --vae-tiling: sd.cpp master-1015's spatially tiled
			// LTX video decode stopped progressing on ROCm (the R9700 sat
			// idle in it for two hours). Tiling only in time, 121 frames of
			// 1280×736 peaked at 29.6 GiB under a 28.4 GiB limit, against
			// 30.9 GiB untiled, so Full fits a 32 GB card; the decode took
			// 38 s against 28 s, with no seams between the tiles.
			LaunchArgs:   []string{"--steps", "8", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--temporal-tiling", "--backend", "te=cpu"},
			Capabilities: []string{CapText, CapStartImage, CapEndImage},
			License:      "LTX-2 Community Licence (free under US$10M yearly revenue); text model: Gemma Terms of Use",
			Notes: "Text-to-video, image-to-video and start-and-end videos with sound; 8 steps. " +
				"WebM videos have a sound track; animated WebP is silent. Up to 5 seconds at 1280×720. " +
				"On a Radeon AI PRO R9700 (32 GB), about 35 s for the 33-frame 768×512 Short size, " +
				"and about 4.5 minutes for 5 s at 1280×720 (reservation 27.7 GiB, measured at Full, with its text model on the CPU); with a start and an end image, about 11 minutes. " +
				"sd-server makes 1280×720 as 1280×736. On a 16 GB RTX A4000, Measure works at Short (14.4 GiB, about 140 s per video), " +
				"but stable-diffusion.cpp crashed there with a CUDA error in one of two videos. " +
				"Licence: free unless the organisation makes US$10 million a year or more; terms at https://huggingface.co/Lightricks/LTX-2.3. " +
				"Its Gemma 3 text model is under Google's Gemma Terms of Use.",
			// 121 frames is 5 s at 24 fps.
			DefaultWidth:  768,
			DefaultHeight: 512,
			DefaultFrames: 33,
			DefaultFPS:    24,
			MaxFrames:     121,
			MaxWidth:      1280,
			MaxHeight:     720,
			VideoSizes: []VideoSize{
				{Label: "Short", Frames: 33, Width: 768, Height: 512},
				{Label: "Medium", Frames: 65, Width: 1280, Height: 720},
				{Label: "Full", Frames: 121, Width: 1280, Height: 720},
			},
			// Measured 2026-10-29 with the one-step Measure and the text
			// model on the CPU: Full granted in 123 s, peak 26.3 GiB.
			FullMeasureTime: "On a Radeon AI PRO R9700, measuring Full took about 2 minutes.",
			// The LTX VAE scales by 32.
			SizeMultiple: 32,
		},
		hunyuanEntry("hunyuan15-i2v-480p-step-distilled", "HunyuanVideo 1.5 I2V 480p step-distilled", "hunyuan-video-15-i2v",
			// No GGUF of the step-distilled model exists. The scaled fp8 file
			// (8.3 GB) made the same video as the fp16 one (16.7 GB) and as
			// the fp16 one converted to Q8_0 as it loaded, with the least
			// VRAM (24.3 GiB against 29.2 and 24.8) and the quickest load
			// (5.6 s against 23.5 and 9.5); R9700, 2026-10-29.
			Component{Slot: "diffusion_model", Repo: hunyuanRepo, File: "split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp8_scaled.safetensors", Bytes: 8335127098},
			false, "8", "1", "7",
			"Image-to-video at 848×480 in 8 steps: the fastest HunyuanVideo 1.5 set (Tencent suggests 8 or 12 steps). Needs a start image. "+
				hunyuan480Fit+" A 33-frame video took about 2 minutes.",
			"On a Radeon AI PRO R9700, measuring Full took about 4 minutes."),
		hunyuanEntry("hunyuan15-i2v-480p-cfg-distilled", "HunyuanVideo 1.5 I2V 480p CFG-distilled (Q8)", "hunyuan-video-15-i2v",
			hunyuanGGUF("I2V_480p", "480p_distilled/hunyuanvideo1.5_480p_i2v_cfg_distilled-Q8_0.gguf"), false, "50", "1", "5",
			"Image-to-video at 848×480, 50 steps. CFG-distilled: one pass per step instead of two, so about twice as fast as the base 480p I2V set. Needs a start image. "+
				hunyuan480Fit+" A 33-frame video took about 8 minutes.",
			"On a Radeon AI PRO R9700, measuring Full took about 4 minutes."),
		hunyuanEntry("hunyuan15-i2v-480p", "HunyuanVideo 1.5 I2V 480p (Q8)", "hunyuan-video-15-i2v",
			hunyuanGGUF("I2V_480p", "480p/hunyuanvideo1.5_480p_i2v-Q8_0.gguf"), false, "50", "6", "5",
			"Image-to-video at 848×480, 50 steps: Tencent's base 480p model. Needs a start image. "+
				hunyuan480Fit+" A 33-frame video took about 16 minutes.",
			"On a Radeon AI PRO R9700, measuring Full took about 6 minutes."),
		hunyuanEntry("hunyuan15-i2v-720p-cfg-distilled", "HunyuanVideo 1.5 I2V 720p CFG-distilled (Q8)", "hunyuan-video-15-i2v",
			hunyuanGGUF("I2V_720p", "720p_distilled/hunyuanvideo1.5_720p_i2v_cfg_distilled-Q8_0.gguf"), true, "50", "1", "7",
			"Image-to-video at 1280×720, 50 steps. CFG-distilled: one pass per step instead of two, so about twice as fast as the base 720p I2V set. Needs a start image. "+
				hunyuan720Fit+" A 33-frame video took about 37 minutes.",
			hunyuan720Measure),
		hunyuanEntry("hunyuan15-i2v-720p", "HunyuanVideo 1.5 I2V 720p (Q8)", "hunyuan-video-15-i2v",
			hunyuanGGUF("I2V_720p", "720p/hunyuanvideo1.5_720p_i2v-Q8_0.gguf"), true, "50", "6", "7",
			"Image-to-video at 1280×720, 50 steps: Tencent's base 720p model, the most detail. Needs a start image. "+
				hunyuan720Fit+" Not timed; expect about twice the CFG-distilled 720p set's 37 minutes for 33 frames.",
			hunyuan720Measure),
		hunyuanEntry("hunyuan15-t2v-480p-cfg-distilled", "HunyuanVideo 1.5 T2V 480p CFG-distilled (Q8)", "hunyuan-video-15-t2v",
			hunyuanGGUF("T2V_480p", "480p_distilled/hunyuanvideo1.5_480p_t2v_cfg_distilled-Q8_0.gguf"), false, "50", "1", "5",
			"Text-to-video at 848×480, 50 steps. CFG-distilled: one pass per step instead of two, so about twice as fast as the base 480p T2V set. "+
				hunyuan480Fit+" A 33-frame video took about 8 minutes.",
			"On a Radeon AI PRO R9700, measuring Full took about 4 minutes."),
		hunyuanEntry("hunyuan15-t2v-480p", "HunyuanVideo 1.5 T2V 480p (Q8)", "hunyuan-video-15-t2v",
			hunyuanGGUF("T2V_480p", "480p/hunyuanvideo1.5_480p_t2v-Q8_0.gguf"), false, "50", "6", "5",
			"Text-to-video at 848×480, 50 steps: Tencent's base 480p model. "+
				hunyuan480Fit+" A 33-frame video took about 16 minutes.",
			"On a Radeon AI PRO R9700, measuring Full took about 6 minutes."),
		hunyuanEntry("hunyuan15-t2v-720p", "HunyuanVideo 1.5 T2V 720p (Q8)", "hunyuan-video-15-t2v",
			hunyuanGGUF("T2V_720p", "720p/hunyuanvideo1.5_720p_t2v-Q8_0.gguf"), true, "50", "6", "9",
			"Text-to-video at 1280×720, 50 steps: Tencent's base 720p model, the most detail. (Tencent has not released a CFG-distilled 720p T2V model yet.) "+
				hunyuan720Fit+" A 33-frame video took about 70 minutes.",
			hunyuan720Measure),
		{
			ID:         "ace-step-turbo",
			Name:       "ACE-Step 1.5 turbo",
			Kind:       KindAce,
			Components: aceComponents("acestep-v15-turbo-Q8_0.gguf", 2549528000),
			License:    "MIT",
			Notes:      "Songs with vocals and lyrics, 8 steps. Shares three files with ACE-Step 1.5 XL SFT.",
		},
		{
			ID:         "ace-step-xl-sft",
			Name:       "ACE-Step 1.5 XL SFT",
			Kind:       KindAce,
			Components: aceComponents("acestep-v15-xl-sft-Q8_0.gguf", 5305828704),
			License:    "MIT",
			Notes:      "Higher quality, 50 steps; a 60-second song in about 17–19 s on a Radeon AI PRO R9700, peaking at about 7 GB of VRAM. Can't add, pull out or complete instruments (use XL base). Shares three files with ACE-Step 1.5 turbo.",
		},
		{
			ID:         "ace-step-xl-base",
			Name:       "ACE-Step 1.5 XL base",
			Kind:       KindAce,
			Components: aceComponents("acestep-v15-xl-base-Q8_0.gguf", 5305828704),
			License:    "MIT",
			Notes:      "The one that can add, pull out or complete instruments in a song; 50 steps. Shares three files with the other ACE-Step sets.",
		},
	}
}

// LookupCatalog returns the catalog entry with the given ID.
func LookupCatalog(id string) (CatalogEntry, bool) {
	for _, e := range Catalog() {
		if e.ID == id {
			return e, true
		}
	}
	return CatalogEntry{}, false
}

// MadeSize is the size sd-server makes for a requested width and height:
// each rounded up to the catalog entry's SizeMultiple. It is the catalog's
// current value, so sets downloaded before it was added have it too; a
// custom set, with no catalog entry, keeps the size asked for.
func (m ModelSet) MadeSize(width, height int) (int, int) {
	e, ok := LookupCatalog(m.CatalogID)
	if m.CatalogID == "" || !ok || e.SizeMultiple <= 1 {
		return width, height
	}
	up := func(n int) int { return (n + e.SizeMultiple - 1) / e.SizeMultiple * e.SizeMultiple }
	return up(width), up(height)
}

// RunArgs returns the backend flags a set runs with. A catalog set takes
// its catalog entry's current flags, so a change to the catalog also
// reaches sets downloaded before it; a custom set keeps its own.
func (m ModelSet) RunArgs() []string {
	if m.CatalogID != "" {
		for _, e := range Catalog() {
			if e.ID == m.CatalogID {
				return e.LaunchArgs
			}
		}
	}
	return m.LaunchArgs
}
