package generation

import "strings"

// FlagReasons says, in a sentence or two, why a set runs with a backend
// flag. The key is the flag, or "flag value" when the reason depends on
// the value. A catalog entry's FlagNotes take precedence for reasons that
// only hold for that set. Shown in a set's Details; see FlagReason.
var FlagReasons = map[string]string{
	"--steps":                       "How many denoising steps each image or video takes. More steps give more detail and take longer. The value is the one the model maker suggests.",
	"--cfg-scale":                   "How closely the result follows the prompt. Above 1, each step takes two passes: one with the prompt and one with the negative prompt.",
	"--cfg-scale 1":                 "One pass per step. Distilled and turbo models are trained to work without guidance, so negative prompts do nothing.",
	"--flow-shift":                  "Moves the denoising steps toward the noisy or the clean end of the schedule. The value is the one the model maker suggests.",
	"--sampling-method":             "The sampler that runs the denoising steps.",
	"--sampling-method euler":       "Euler is the sampler these flow-matching models are made for.",
	"--high-noise-steps":            "Steps on the high-noise expert of a Wan2.2 A14B pair, which runs first and sets the motion.",
	"--high-noise-cfg-scale":        "Guidance for the high-noise expert of a Wan2.2 A14B pair.",
	"--high-noise-sampling-method":  "The sampler for the high-noise expert of a Wan2.2 A14B pair.",
	"--moe-boundary":                "Where the high-noise expert hands over to the low-noise expert, as a part of the noise schedule. 0.9 is the Wan2.2 I2V recipe's value; sd-server's own default is 0.875.",
	"--diffusion-fa":                "Flash attention in the diffusion model. It uses less VRAM and is faster.",
	"--vae-tiling":                  "Decodes the image in tiles. This lowers the VAE's peak VRAM use, with no visible seams.",
	"--temporal-tiling":             "Decodes the video a few frames at a time, with each frame whole. This lowers the decode's peak VRAM use. --vae-tiling hung on ROCm with these video VAEs.",
	"--backend":                     "Chooses the device that runs a part of the model.",
	"--backend te=cpu":              "Runs the text model on the CPU. On a Radeon AI PRO R9700 this saved 4 to 6 GiB of VRAM on most sets. It adds a second or two to an image and 13 to 16 s to a video.",
	"--negative-prompt":             "The negative prompt for requests that do not give their own.",
	"--sigmas":                      "A fixed noise schedule in place of the sampler's own.",
	"--sigmas qwen-image-2.1-turbo": "The noise schedule Qwen-Image 2.1 Turbo was distilled with, so 8 steps are enough.",
	"--model-args":                  "Options that sd-server passes to the model.",
	"--model-args qwen_image_zero_cond_t=true": "Needed by Qwen-Image-Edit 2511: stable-diffusion.cpp's docs say edit quality drops a lot without it.",

	// Flags toolchest adds itself (ToolchestFlags).
	"--listen-ip":           "The address sd-server listens on: the same one toolchest's router uses.",
	"--listen-port":         "The instance's own port, from the generation port range.",
	"--host":                "The address ace-server listens on: the same one toolchest's router uses.",
	"--port":                "The instance's own port, from the generation port range.",
	"--max-vram":            "The instance's VRAM reservation, so stable-diffusion.cpp plans its memory inside it. Measure sets the reservation.",
	"--lora-model-dir":      "The folder with the set's LoRA files. Each request names the LoRAs to apply.",
	"--hires-upscalers-dir": "A folder with a link to the set's ESRGAN file. sd-server loads it for each upscale.",
	"--models":              "A folder with links to the set's four files. ace-server loads the models it finds there.",
	"--max-batch":           "ace-server makes one song at a time. Toolchest queues the others.",
}

// ToolchestFlags are the flags toolchest adds to every instance of a kind
// of set, beside the set's own flags and its file flags.
var ToolchestFlags = []string{"--listen-ip", "--listen-port", "--host", "--port", "--max-vram",
	"--lora-model-dir", "--hires-upscalers-dir", "--models", "--max-batch"}

// FlagReason returns why a set runs with a flag: the entry's FlagNotes for
// "flag value", then for the flag, then the same two keys in FlagReasons;
// "" when none has a reason. A custom set passes a zero CatalogEntry.
func FlagReason(flag, value string, entry CatalogEntry) string {
	keys := []string{flag}
	if value != "" {
		keys = []string{flag + " " + value, flag}
	}
	for _, table := range []map[string]string{entry.FlagNotes, FlagReasons} {
		for _, k := range keys {
			if r := table[k]; r != "" {
				return r
			}
		}
	}
	return ""
}

// SplitFlags pairs each flag in args with its value: the next word, when
// that word is not a flag itself. A flag with no value gets "".
func SplitFlags(args []string) [][2]string {
	var out [][2]string
	for i := 0; i < len(args); i++ {
		pair := [2]string{args[i], ""}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "--") {
			pair[1] = args[i+1]
			i++
		}
		out = append(out, pair)
	}
	return out
}
