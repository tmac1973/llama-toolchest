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
// (mtmd_helper_get_capabilities).
func (m *Model) CanChat(cfg *ModelConfig) bool {
	return !m.IsEmbedding() && (!m.Speaks(cfg) || m.HasChatTemplate)
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
// by the same rules llama-server uses before a model is loaded.
func (m *Model) Modalities(cfg *ModelConfig) (input, output []string) {
	p := m.Projector(cfg)
	input = []string{"text"}
	if m.HasBuiltinVision || p.Vision {
		input = append(input, "image")
	}
	if p.Audio {
		input = append(input, "audio")
	}
	if !p.GenAudio || m.HasChatTemplate {
		output = []string{"text"}
	}
	if p.GenAudio {
		output = append(output, "audio")
	}
	return input, output
}
