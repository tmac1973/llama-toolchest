package generation

import "slices"

// The toolchest overlay: what llama-toolchest knows about sd-server
// recipes that upstream's recipe files do not say. ParseRecipes applies
// it to every recipe book, the snapshot and each build's.

// recipeExtraSlots are slots toolchest adds to a recipe. LoRAs are
// toolchest slots, not upstream ones: sd-server gets their folder
// (--lora-model-dir) and each request names them. The Wan2.2 A14B pairs
// take one for each expert, such as the Lightning 4-step LoRAs.
var recipeExtraSlots = map[string][]string{
	"wan22-i2v": {"lora", "high_noise_lora"},
	"wan22-t2v": {"lora", "high_noise_lora"},
}

// recipeLabels are toolchest's slot labels for a recipe, in toolchest's
// words. They come first; upstream's labels fill the rest (see
// SlotLabel). Written from stable-diffusion.cpp's recipe labels at
// master-1015-e41fbcf (2026-11-01).
var recipeLabels = map[string]map[string]string{
	"ming-image":           {"llm": "Ling Mini 2.0 text model", "tokenizer": "Ling tokenizer (tokenizer.json)"},
	"qwen-image-21-turbo":  qwenImage21Labels,
	"qwen-image-21":        qwenImage21Labels,
	"flux2-klein-4b":       {"llm": "Qwen3 4B text model"},
	"z-image-turbo":        {"llm": "Qwen3 4B text model"},
	"z-image":              {"llm": "Qwen3 4B text model"},
	"flux3-klein":          {"llm": "Qwen3-VL 4B text model"},
	"qwen-image-edit-2511": {"llm": "Qwen2.5-VL 7B text model", "llm_vision": "Qwen2.5-VL 7B vision projector (for edits)", "vae": "Qwen-Image VAE"},
	"sdxl-checkpoint":      {"vae": "VAE override"},
	"wan22-ti2v-5b":        {"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.2 VAE"},
	"wan22-i2v":            wanMoELabels,
	"wan22-t2v":            wanMoELabels,
	"wan21-t2v":            {"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE"},
	"wan21-i2v":            {"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE", "clip_vision": "CLIP Vision H"},
	"wan21-flf2v":          {"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE", "clip_vision": "CLIP Vision H"},
	"wan22-s2v":            {"t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE", "audio_encoder": "wav2vec2 audio encoder (Large English)"},
	"ltx23-distilled":      ltx23Labels,
	"ltx23-dev":            ltx23Labels,
	"ltx25-dev":            {"llm": "LTX-2.5 Gemma 4 12B text model (with projection)", "vae": "LTX-2.5 video VAE", "audio_vae": "LTX-2.5 audio VAE"},
	"hunyuan-video-15-t2v": hunyuanLabels,
	"hunyuan-video-15-i2v": hunyuanLabels,
	"lingbot-video-dense":  {"llm": "Qwen3-VL 4B text model", "vae": "Wan2.1 VAE"},
	"lingbot-video-moe":    {"llm": "Qwen3-VL 4B text model", "vae": "Wan2.1 VAE"},
}

var (
	qwenImage21Labels = map[string]string{"llm": "Qwen3-VL 8B text model", "llm_vision": "Qwen3-VL 8B vision projector (for edits)", "vae": "Qwen-Image 2.1 VAE", "tae": "TAEQI2.1"}
	wanMoELabels      = map[string]string{"diffusion_model": "Low-noise expert", "t5xxl": "UMT5-XXL text encoder", "vae": "Wan2.1 VAE"}
	ltx23Labels       = map[string]string{"llm": "Gemma 3 12B text model", "vae": "LTX-2.3 video VAE", "audio_vae": "LTX-2.3 audio VAE"}
	hunyuanLabels     = map[string]string{"llm": "Qwen2.5-VL 7B text model", "t5xxl": "ByT5 Small GlyphXL (for text in the video)", "vae": "HunyuanVideo 1.5 VAE", "tae": "TAEHV 1.5"}
)

// recipeTextStart are the video recipes that start from a prompt alone
// as well as from an image. A video recipe without init_image always
// starts from text; one with it starts from text only when listed here.
var recipeTextStart = map[string]bool{
	"wan22-ti2v-5b":       true,
	"ltx23-distilled":     true,
	"ltx23-dev":           true,
	"ltx25-dev":           true,
	"lingbot-video-dense": true,
	"lingbot-video-moe":   true,
}

// recipeAlpha are the image recipes that make images with transparency.
var recipeAlpha = map[string]bool{"ming-image": true}

// RecipeMaxEditImages is how many reference images one edit can take, by
// recipe: the model makers' own limits. FLUX.3 Image Klein takes up to
// 10, FLUX.2 klein 4B up to 4 (Black Forest Labs' FLUX.2 specifications)
// and Qwen-Image-Edit 2511 1 to 3. The Qwen-Image 2.1 recipes are left
// out: their limit for Turbo is not published. Measure tests an edit with
// one reference, so many more can need more VRAM than the reservation.
var RecipeMaxEditImages = map[string]int{
	"flux3-klein":          10,
	"flux2-klein-4b":       4,
	"qwen-image-edit-2511": 3,
}

// recipeLeftOut are recipes toolchest leaves out on purpose, with the
// reason shown in the custom set form.
var recipeLeftOut = map[string]string{
	"wan21-vace": "needs control video frames, which llama-toolchest does not send",
}

// recipeAudioInput are the recipes whose ref_audios toolchest sends: one
// track, the video job's audio. Other recipes with that feature stay
// unavailable (recipeNeedsInput).
var recipeAudioInput = map[string]bool{"wan22-s2v": true}

// recipeNeedsInput are recipe features that need an input toolchest does
// not send, with the words for the reason ("needs <what> input").
var recipeNeedsInput = map[string]string{
	"control_image":    "a control image",
	"control_frames":   "a control video",
	"vace":             "a control video",
	"ref_audios":       "an audio",
	"ref_videos":       "a reference video",
	"ref_video_audios": "a reference video",
}

// applyOverlay adds toolchest's extra slots to a parsed recipe.
func applyOverlay(r *Recipe) {
	for _, s := range recipeExtraSlots[r.ID] {
		if !slices.Contains(r.Slots.Slots(), s) {
			r.Slots.Optional = append(r.Slots.Optional, s)
		}
	}
}
