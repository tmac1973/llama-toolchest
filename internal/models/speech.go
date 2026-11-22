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
// answered /tts with a copied voice (issue #680).
//
// Qwen3-TTS was measured on 2026-11-22 on a Radeon AI PRO R9700 with
// llama.cpp b12634, at the default settings (8192 context, 4 slots,
// flash attention): 5055 MiB loaded, 5206 MiB after requests, with and
// without speaker_ref_text. Toolchest's estimate for the same settings is
// 6.3 GB, so its load checks keep enough room. A 4.9 s clip was enough to
// copy a voice. The same seed gave another take on the GPU (with and
// without flash attention, one slot or four) and the same audio on the
// CPU; another seed always gave another take.
//
// Qwen3-TTS comes from moonbox-dev, not ggml-org: ggml-org's projector
// (August 2026) predates the reference encoder (llama.cpp #32438, b12491),
// so speaker_ref_text does nothing with it and the voice is copied from
// the speaker embedding only. moonbox-dev's has the encoder.
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
	}
}
