package models

import "fmt"

// SpeechModelPreset defines a curated text-to-speech model available for
// one-click download. A speech model needs two files from its repo: the
// model and its projector, which holds the speaker encoder, the audio
// code predictor and the decoder. Both land in the repo's folder, where
// the model finds its projector.
type SpeechModelPreset struct {
	Name           string `json:"name"`
	Repo           string `json:"repo"`            // HuggingFace repo ID
	Filename       string `json:"filename"`        // model GGUF in the repo
	MmprojFilename string `json:"mmproj_filename"` // projector GGUF in the repo
	SizeMB         int    `json:"size_mb"`         // both files together
	// MinBuild is the oldest llama.cpp build (the N of bN) with everything
	// the card offers.
	MinBuild int `json:"min_build"`
	// VRAMMiB is the GPU memory measured with the model loaded at its
	// default settings and one speech request answered.
	VRAMMiB   int    `json:"vram_mib"`
	Languages string `json:"languages"`
	License   string `json:"license"`
	Desc      string `json:"desc"` // Short description
	// Voice says how to give the model a voice to copy.
	Voice string `json:"voice"`
}

// SizeLabel is the download size for the card: MB below 1 GB, GB above.
func (p SpeechModelPreset) SizeLabel() string { return sizeLabelMB(p.SizeMB) }

// VRAMLabel is the measured GPU memory for the card.
func (p SpeechModelPreset) VRAMLabel() string {
	return fmt.Sprintf("%.1f GiB", float64(p.VRAMMiB)/1024)
}

// CuratedSpeechModels returns the text-to-speech models offered on the
// Download Models tab. Every entry was checked against its files and
// answered /tts in each of the ways its card names (issue #680).
//
// Qwen3-TTS was measured on 2026-11-22 on a Radeon AI PRO R9700 with
// llama.cpp b12634, at the default settings (8192 context, 4 slots,
// flash attention): 5055 MiB loaded, 5206 MiB after requests, with and
// without speaker_ref_text. A 4.9 s clip was enough to copy a voice. The same seed gave another take on the GPU (with and
// without flash attention, one slot or four) and the same audio on the
// CPU; another seed always gave another take.
//
// Qwen3-TTS comes from moonbox-dev, not ggml-org: ggml-org's projector
// (August 2026) predates the reference encoder (llama.cpp #32438, b12491),
// so speaker_ref_text does nothing with it and the voice is copied from
// the speaker embedding only. moonbox-dev's has the encoder.
//
// The other five were measured on 2026-11-23 the same way (R9700, b12634,
// 8192 context, 4 slots, flash attention), with a 5 s clip made by Kokoro
// as the voice to copy. VRAM is the most seen after the requests:
//
//	Kokoro 82M         384 MiB loaded,  695 after; 54 voices; speed works;
//	                   speaker_ref answers 400 (no voice encoder)
//	Chatterbox Turbo  2060 MiB loaded, 2201 after; voice "default" or a
//	                   copied one; speed answers 400
//	ZONOS3 Mini       2384 MiB loaded, 2585 after; 27 voices, a copied
//	                   voice, speed and loudness all work
//	VoxCPM2           4819 MiB loaded, 5020 after; a described voice, a
//	                   copied one, flow_steps and cfg_scale all work
//	Fish S2 Pro       8405 MiB loaded, 9096 after; speaker tags and
//	                   [laughing]; a copied voice with and without
//	                   speaker_ref_text
//
// Toolchest's estimate counts a projector at its file size, which is
// short of what a speech generator takes; SpeechGenScratchGiB adds the
// difference, so the load checks keep enough room for each of them.
//
// VoxCPM2 and S2 Pro carry the chat template of their text model, and
// llama-server answers /v1/chat/completions for them with noise or
// nothing; toolchest treats them as speech-only (ProjectorCaps.SpeechOnly).
func CuratedSpeechModels() []SpeechModelPreset {
	return []SpeechModelPreset{
		{
			Name:           "Qwen3-TTS 1.7B Base",
			Repo:           "moonbox-dev/Qwen3-TTS-12Hz-1.7B-Base-GGUF",
			Filename:       "qwen3-tts-1-7b-Q8_0.gguf",
			MmprojFilename: "mmproj-qwen3-tts-1-7b-F16.gguf",
			SizeMB:         2688,
			MinBuild:       12491,
			VRAMMiB:        5206,
			Languages:      "en, zh, de, fr, es, it, pt, ru, ja, ko",
			License:        "Apache-2.0",
			Desc:           "Qwen, 2026. Copies a voice from a short clip, 24 kHz, 1.7B params",
			Voice:          "Send a clip of the voice as speaker_ref. With its transcript in speaker_ref_text the copy is closer; the clip must then be 30 s or shorter. A clean clip of about 5 s of one speaker is enough.",
		},
		{
			Name:           "Kokoro 82M",
			Repo:           "ggml-org/Kokoro-82M-GGUF",
			Filename:       "Kokoro-82M-F32.gguf",
			MmprojFilename: "mmproj-Kokoro-82M-F16.gguf",
			SizeMB:         270,
			MinBuild:       12494,
			VRAMMiB:        695,
			Languages:      "en (others as phonemes)",
			License:        "Apache-2.0",
			Desc:           "hexgrad, 2025. Small and fast, 54 voices of its own, 24 kHz, 82M params",
			Voice:          "Pick one of its voices with voice (af_heart by default; the first letter is the language). It cannot copy a voice. It turns English text into phonemes itself; for other languages send phonemes with raw_prompt. Takes speed.",
		},
		{
			Name:           "Chatterbox Turbo",
			Repo:           "ggml-org/chatterbox-turbo-GGUF",
			Filename:       "chatterbox-turbo-Q8_0.gguf",
			MmprojFilename: "mmproj-chatterbox-turbo-F16.gguf",
			SizeMB:         1037,
			MinBuild:       12159,
			VRAMMiB:        2201,
			Languages:      "en",
			License:        "MIT",
			Desc:           "Resemble AI, 2025. English, copies a voice from a short clip, 24 kHz, 350M params",
			Voice:          "Send a clip of the voice as speaker_ref. Without one it speaks in its own voice.",
		},
		{
			Name:           "ZONOS3 Mini",
			Repo:           "ggml-org/ZONOS3-Mini-GGUF",
			Filename:       "ZONOS3-Mini-Q8_0.gguf",
			MmprojFilename: "mmproj-ZONOS3-Mini-F32.gguf",
			SizeMB:         1997,
			MinBuild:       12424,
			VRAMMiB:        2585,
			Languages:      "43, such as en, de, fr, es, it, pt, zh, ja, ko",
			License:        "Apache-2.0",
			Desc:           "Zyphra, 2026. Voices of its own or a copied voice, 44.1 kHz, 800M params (100M active)",
			Voice:          "Pick one of its voices with voice (llama-server's GET /props lists them), or send a clip of the voice as speaker_ref (the first 60 s are used). Takes speed and loudness.",
		},
		{
			Name:           "VoxCPM2",
			Repo:           "ggml-org/VoxCPM2-GGUF",
			Filename:       "VoxCPM2-Q8_0.gguf",
			MmprojFilename: "mmproj-VoxCPM2-BF16.gguf",
			SizeMB:         3574,
			MinBuild:       12531,
			VRAMMiB:        5020,
			Languages:      "30, such as en, zh, de, fr, es, ja, ko, ar",
			License:        "Apache-2.0",
			Desc:           "OpenBMB, 2026. Makes a voice from a description, or copies one, 48 kHz, 2B params",
			Voice:          "Describe the voice in brackets at the start of the text, such as (A calm male narrator), or send a clip of the voice as speaker_ref (no transcript needed, the first 30 s are used).",
		},
		{
			Name:           "Fish Audio S2 Pro",
			Repo:           "ggml-org/s2-pro-GGUF",
			Filename:       "s2-pro-Q8_0.gguf",
			MmprojFilename: "mmproj-s2-pro-BF16.gguf",
			SizeMB:         6337,
			MinBuild:       12452,
			VRAMMiB:        9096,
			Languages:      "80+, such as en, zh, ja, ko, de, fr, es",
			License:        "Fish Audio Research (non-commercial)",
			Desc:           "Fish Audio, 2026. Expressive: tags such as [laughing] or [whisper], several speakers in one text, 44.1 kHz, 4B params",
			Voice:          "Send a clip of the voice as speaker_ref, and what is said in it as speaker_ref_text for a closer copy (the first 30 s are used). Give each speaker a tag, <|speaker:0|>, <|speaker:1|>.",
		},
	}
}
