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
	// DefaultWidth and DefaultHeight are the image size the measured test
	// run uses; zero for ace.
	DefaultWidth  int `json:"default_width,omitempty"`
	DefaultHeight int `json:"default_height,omitempty"`
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

// Files shared by more than one image set. A shared file downloads once.
var (
	// fluxAE is the FLUX.1 VAE that Z-Image and Chroma use. The copy in
	// black-forest-labs/FLUX.1-schnell is gated; this one is not.
	fluxAE = Component{Slot: "vae", Repo: "Comfy-Org/z_image_turbo", File: "split_files/vae/ae.safetensors", Bytes: 335304388}
	// flux2SmallDecoder is a FLUX.2 VAE, used by FLUX.2 klein and FLUX.3
	// Image Klein. The FLUX.2-dev VAE is gated; this one is ungated and
	// Apache-licensed.
	flux2SmallDecoder = Component{Slot: "vae", Repo: "black-forest-labs/FLUX.2-small-decoder", File: "full_encoder_small_decoder.safetensors", Bytes: 249519092}
)

// Catalog returns the built-in model sets. File sizes were read from the
// Hugging Face tree API on 2026-10-24 (2026-10-25 for Z-Image Turbo, FLUX.3
// Image Klein, Chroma1-HD and Qwen-Image-Edit 2511); launch settings for
// Qwen-Image 2.1 Turbo and ACE-Step XL SFT are the ones measured on
// asmodean on 2026-10-24.
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
			LaunchArgs: []string{"--sigmas", "qwen-image-2.1-turbo", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling"},
			// No "edit": Qwen-Image edits need the llm_vision weights,
			// which this set leaves out.
			License:       "Qwen-Image 2.1 licence",
			Notes:         "8 steps; about 36 s per 1024×1024 image on a Radeon AI PRO R9700, using about 10 GB of VRAM. Licence terms: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
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
			LaunchArgs:    []string{"--steps", "4", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa"},
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
			LaunchArgs:    []string{"--steps", "9", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling"},
			License:       "Apache-2.0",
			Notes:         "9 steps; about 19 s per 1024×1024 image on a Radeon AI PRO R9700, using about 11 GB of VRAM. Shares its VAE with Chroma1-HD.",
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
			LaunchArgs:    []string{"--steps", "16", "--cfg-scale", "1", "--sampling-method", "euler", "--diffusion-fa"},
			Capabilities:  []string{"edit"},
			License:       "FLUX Kommunity License (non-commercial)",
			Notes:         "16 steps; about 21 s per 1024×1024 image (56 s for an edit) on a Radeon AI PRO R9700, using about 12 GB of VRAM. Also edits images, with up to 10 reference images. Shares its VAE with FLUX.2 klein 4B. Non-commercial licence: only people and companies with less than US$5 million yearly revenue may use the images commercially. Licence terms: https://huggingface.co/black-forest-labs/flux-3-image-klein",
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
				{Slot: "t5xxl", Repo: "city96/t5-v1_1-xxl-encoder-gguf", File: "t5-v1_1-xxl-encoder-Q8_0.gguf", Bytes: 5061584064},
				fluxAE,
			},
			// Without a negative prompt, Chroma's images came out covered
			// in grainy noise on 2026-10-25; 20 steps still left some, 40
			// did not. --negative-prompt sets the default for requests that
			// give none.
			LaunchArgs: []string{"--steps", "40", "--cfg-scale", "4", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling",
				"--negative-prompt", "low quality, ugly, unfinished, out of focus, deformed, disfigured, blurry, smudged, restricted palette, flat colors"},
			License:       "Apache-2.0",
			Notes:         "40 steps; about 3.5 minutes per 1024×1024 image on a Radeon AI PRO R9700, using about 15 GB of VRAM. Based on FLUX.1-schnell. Starts with a default negative prompt; a request can give its own. Shares its VAE with Z-Image Turbo.",
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
				{Slot: "llm", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", File: "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf", Bytes: 4683072512},
				{Slot: "llm_vision", Repo: "mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", File: "Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf", Bytes: 1354162912},
				{Slot: "vae", Repo: "Comfy-Org/Qwen-Image_ComfyUI", File: "split_files/vae/qwen_image_vae.safetensors", Bytes: 253806246},
			},
			// Without qwen_image_zero_cond_t=true, sd.cpp's docs say edit
			// quality drops a lot.
			LaunchArgs:    []string{"--steps", "20", "--cfg-scale", "4", "--flow-shift", "3", "--sampling-method", "euler", "--diffusion-fa", "--vae-tiling", "--model-args", "qwen_image_zero_cond_t=true"},
			Capabilities:  []string{"edit"},
			License:       "Apache-2.0",
			Notes:         "Edits images from one or more reference images; 20 steps. About 6 minutes per 1024×1024 edit on a Radeon AI PRO R9700, using about 21 GB of VRAM, so it needs a 24 GB card.",
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
