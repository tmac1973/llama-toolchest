package models

import (
	"os"
	"sync"
	"time"
)

// ProjectorCaps is what a projector (mmproj) file adds to a model, as
// llama.cpp reads it from the file's clip.has_*_encoder keys.
type ProjectorCaps struct {
	Vision   bool // reads images (clip.has_vision_encoder)
	Audio    bool // reads audio (clip.has_audio_encoder)
	GenAudio bool // speaks: makes audio (clip.has_gen_audio_encoder)
	// GenAudioType is clip.gen.audio.projector_type: which audio
	// generator the projector holds (qwen3tts_gen, kokoro, lfm3a_gen...).
	GenAudioType string
	// keys is whether the file had any of the keys. A projector without
	// them is from before audio projectors existed, so it is for images.
	keys bool
}

func (c *ProjectorCaps) set(key string, v bool) {
	c.keys = true
	switch key {
	case "clip.has_vision_encoder":
		c.Vision = v
	case "clip.has_audio_encoder":
		c.Audio = v
	case "clip.has_gen_audio_encoder":
		c.GenAudio = v
	}
}

// speechOnlyGenTypes are the audio generators of models that only speak.
// Some of them come with the chat template of the text model they are
// built on (VoxCPM2 has MiniCPM's, Fish S2 Pro has Qwen3's), so the
// template does not tell them from a model that also chats. llama.cpp's
// own rule (mtmd_helper_get_capabilities) goes by the template alone and
// reports those two as chat models; toolchest does not offer them for
// chat. Not listed, because the model also answers in text: lfm3a_gen
// (LFM3-Audio) and moshi_gen (Hibiki, realtime only).
var speechOnlyGenTypes = map[string]bool{
	"qwen3tts_gen":       true,
	"pockettts_gen":      true,
	"soprano":            true,
	"kokoro":             true,
	"tiny_voca":          true,
	"chatterbox":         true,
	"drama3_gen":         true,
	"zonos3_gen":         true,
	"fish_s2_gen":        true,
	"voxcpm2_gen":        true,
	"yue2_gen":           true,
	"minimax_music3_gen": true,
}

// speechGenScratchGiB is the GPU memory a speech generator takes beyond
// its projector file: the work buffers of its code predictor and audio
// decoder, which grow during the first requests. The estimate counts the
// projector at its file size only (AuxFilesVRAMGB), which left a speech
// model up to 1.5 GiB short. Each figure is what was measured on a Radeon
// AI PRO R9700 (llama.cpp b12634, 8192 context, 4 slots, flash attention,
// after /tts requests) minus the estimate for the same settings, plus
// 0.2 GiB of room, rounded up to 0.1 GiB (2026-11-23):
//
//	Qwen3-TTS 1.7B Q8_0    5206 MiB measured, 3.6 GiB estimated
//	Fish S2 Pro Q8_0       9096 MiB measured, 7.6 GiB estimated
//	VoxCPM2 Q8_0           5020 MiB measured, 4.1 GiB estimated
//	ZONOS3 Mini Q8_0       2585 MiB measured, 2.4 GiB estimated
//
// Chatterbox Turbo (2201 MiB, 2.2 GiB estimated) and Kokoro (695 MiB,
// 0.8 GiB) need nothing more. A generator not listed adds nothing.
var speechGenScratchGiB = map[string]float64{
	"qwen3tts_gen": 1.7,
	"fish_s2_gen":  1.5,
	"voxcpm2_gen":  1.0,
	"zonos3_gen":   0.4,
}

// SpeechGenScratchGiB is the GPU memory the config's speech generator
// takes beyond its projector file (see speechGenScratchGiB): 0 when the
// projector is not loaded, is kept in system memory, or does not speak.
func SpeechGenScratchGiB(cfg *ModelConfig) float64 {
	if cfg == nil || !ProjectorLoaded(cfg) || cfg.MmprojInRAM {
		return 0
	}
	p := ReadProjectorCaps(cfg.MmprojPath)
	if !p.GenAudio {
		return 0
	}
	return speechGenScratchGiB[p.GenAudioType]
}

// SpeechOnly reports whether the projector's audio generator belongs to a
// model that only speaks, whatever chat template the model has.
func (c ProjectorCaps) SpeechOnly() bool {
	return c.GenAudio && speechOnlyGenTypes[c.GenAudioType]
}

// projectorCache keeps each projector file's capabilities, so the model
// list does not read every projector on each request. An entry is used
// while the file's size and time are unchanged.
var projectorCache = struct {
	sync.Mutex
	m map[string]projectorEntry
}{m: map[string]projectorEntry{}}

type projectorEntry struct {
	size int64
	mod  time.Time
	caps ProjectorCaps
}

// ReadProjectorCaps reads what the projector file at path can do. A file
// that cannot be read, or that has none of the keys, is taken to be an
// image projector, which is what toolchest assumed for every projector
// before audio ones existed.
func ReadProjectorCaps(path string) ProjectorCaps {
	imageOnly := ProjectorCaps{Vision: true}
	st, err := os.Stat(path)
	if err != nil {
		return imageOnly
	}
	projectorCache.Lock()
	e, ok := projectorCache.m[path]
	projectorCache.Unlock()
	if ok && e.size == st.Size() && e.mod.Equal(st.ModTime()) {
		return e.caps
	}
	caps := imageOnly
	if meta, err := ParseGGUFMeta(path); err == nil && meta.Projector.keys {
		caps = meta.Projector
	}
	projectorCache.Lock()
	projectorCache.m[path] = projectorEntry{st.Size(), st.ModTime(), caps}
	projectorCache.Unlock()
	return caps
}

// Projector returns what the model's projector adds: nothing when it has
// none, or when it is set but not loaded (MmprojDisabled).
func (m *Model) Projector(cfg *ModelConfig) ProjectorCaps {
	if cfg == nil || cfg.MmprojPath == "" || cfg.MmprojDisabled {
		return ProjectorCaps{}
	}
	return ReadProjectorCaps(cfg.MmprojPath)
}

// HasVision reports whether the model can read images: built in, or
// through its projector.
func (m *Model) HasVision(cfg *ModelConfig) bool {
	return m.HasBuiltinVision || m.Projector(cfg).Vision
}

// Speaks reports whether the model can make speech, served by llama-server's
// /tts and /v1/realtime.
func (m *Model) Speaks(cfg *ModelConfig) bool {
	return m.Projector(cfg).GenAudio
}

// CanChat reports whether the model answers in text. A model that speaks
// can chat only when it also has a chat template; one without is
// text-to-speech only. llama.cpp uses the same rule
// (mtmd_helper_get_capabilities), except that toolchest also knows the
// speech-only generators that come with a chat template (SpeechOnly).
func (m *Model) CanChat(cfg *ModelConfig) bool {
	return m.Kind() == KindChat && m.answersInText(m.Projector(cfg))
}

// answersInText is whether a model with projector p makes text: always
// when it does not speak, else only with a chat template and a generator
// that is not speech-only.
func (m *Model) answersInText(p ProjectorCaps) bool {
	return !p.GenAudio || (m.HasChatTemplate && !p.SpeechOnly())
}

// UnderstandsAudio reports whether the model understands speech: it reads
// audio and answers it (transcription, voice chat). A text-to-speech model
// also reads audio, but only as a voice sample to copy, so it does not
// count.
func (m *Model) UnderstandsAudio(cfg *ModelConfig) bool {
	return m.Projector(cfg).Audio && m.CanChat(cfg)
}

// Modalities returns what the model reads and makes, in the shape of
// llama-server's architecture.input_modalities and output_modalities, and
// by the same rules llama-server uses before a model is loaded (and
// SpeechOnly, see CanChat).
func (m *Model) Modalities(cfg *ModelConfig) (input, output []string) {
	p := m.Projector(cfg)
	input = []string{"text"}
	if m.HasBuiltinVision || p.Vision {
		input = append(input, "image")
	}
	if p.Audio {
		input = append(input, "audio")
	}
	if m.answersInText(p) {
		output = []string{"text"}
	}
	if p.GenAudio {
		output = append(output, "audio")
	}
	return input, output
}
