package models

import (
	"os"
	"slices"
	"testing"
	"time"
)

// writeProjector writes a projector (mmproj) file with the given
// clip.has_*_encoder keys, as llama.cpp's converter does.
func writeProjector(t *testing.T, vision, audio, genAudio *bool) string {
	t.Helper()
	kvs := []testKV{
		{"general.architecture", ggufTypeString, "clip"},
		{"general.type", ggufTypeString, "mmproj"},
	}
	for _, k := range []struct {
		key string
		v   *bool
	}{{"clip.has_vision_encoder", vision}, {"clip.has_audio_encoder", audio}, {"clip.has_gen_audio_encoder", genAudio}} {
		if k.v != nil {
			var b uint8
			if *k.v {
				b = 1
			}
			kvs = append(kvs, testKV{k.key, ggufTypeBool, b})
		}
	}
	kvs = append(kvs, testKV{"clip.audio.num_mel_bins", ggufTypeUint32, uint32(128)})
	return writeTestGGUF(t, kvs)
}

func yes() *bool { b := true; return &b }

// writeSpeechProjector writes the projector of a model that speaks and
// takes a voice sample, with its generator type.
func writeSpeechProjector(t *testing.T, genType string) string {
	t.Helper()
	return writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "clip"},
		{"general.type", ggufTypeString, "mmproj"},
		{"clip.has_gen_audio_encoder", ggufTypeBool, uint8(1)},
		{"clip.gen.audio.projector_type", ggufTypeString, genType},
		{"clip.has_audio_encoder", ggufTypeBool, uint8(1)},
		{"clip.audio.projector_type", ggufTypeString, genType + "_spkenc"},
	})
}

// What each kind of model reports, against what llama-server says for
// the same files (Qwen3-TTS, LFM3-Audio, Gemma 4, a vision model).
func TestProjectorCapabilities(t *testing.T) {
	cases := []struct {
		name         string
		proj         string
		chatTemplate bool
		caps         []string
		in, out      []string
	}{
		{"text-to-speech only (Qwen3-TTS)", writeProjector(t, nil, yes(), yes()), false,
			[]string{"speech"}, []string{"text", "audio"}, []string{"audio"}},
		{"voice chat (LFM3-Audio)", writeProjector(t, nil, yes(), yes()), true,
			[]string{"chat", "audio", "speech"}, []string{"text", "audio"}, []string{"text", "audio"}},
		{"voice chat with its generator type (LFM3-Audio)", writeSpeechProjector(t, "lfm3a_gen"), true,
			[]string{"chat", "audio", "speech"}, []string{"text", "audio"}, []string{"text", "audio"}},
		// VoxCPM2 and Fish S2 Pro keep the chat template of the text model
		// they are built on. llama-server reports them as chat models;
		// toolchest does not.
		{"text-to-speech only with a chat template (VoxCPM2)", writeSpeechProjector(t, "voxcpm2_gen"), true,
			[]string{"speech"}, []string{"text", "audio"}, []string{"audio"}},
		{"text-to-speech only with a chat template (Fish S2 Pro)", writeSpeechProjector(t, "fish_s2_gen"), true,
			[]string{"speech"}, []string{"text", "audio"}, []string{"audio"}},
		{"images and audio in (Gemma 4)", writeProjector(t, yes(), yes(), nil), true,
			[]string{"chat", "vision", "audio"}, []string{"text", "image", "audio"}, []string{"text"}},
		{"images in", writeProjector(t, yes(), nil, nil), true,
			[]string{"chat", "vision"}, []string{"text", "image"}, []string{"text"}},
		{"an old projector without the keys is for images", writeTestGGUF(t, []testKV{{"general.architecture", ggufTypeString, "clip"}}), true,
			[]string{"chat", "vision"}, []string{"text", "image"}, []string{"text"}},
		{"a projector that cannot be read is taken as for images", "/nonexistent/mmproj.gguf", true,
			[]string{"chat", "vision"}, []string{"text", "image"}, []string{"text"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := &Model{ID: "m", HasChatTemplate: c.chatTemplate}
			cfg := &ModelConfig{MmprojPath: c.proj}
			if got := m.Capabilities(cfg); !slices.Equal(got, c.caps) {
				t.Errorf("capabilities = %v, want %v", got, c.caps)
			}
			in, out := m.Modalities(cfg)
			if !slices.Equal(in, c.in) || !slices.Equal(out, c.out) {
				t.Errorf("modalities = %v / %v, want %v / %v", in, out, c.in, c.out)
			}
		})
	}
}

// A projector that is set but not loaded adds nothing, and a model
// without one is a plain chat model.
func TestProjectorNotLoaded(t *testing.T) {
	m := &Model{ID: "m"}
	proj := writeProjector(t, nil, yes(), yes())
	for _, cfg := range []*ModelConfig{nil, {}, {MmprojPath: proj, MmprojDisabled: true}} {
		if got := m.Capabilities(cfg); !slices.Equal(got, []string{"chat"}) {
			t.Errorf("cfg %+v: capabilities = %v, want [chat]", cfg, got)
		}
		if in, out := m.Modalities(cfg); !slices.Equal(in, []string{"text"}) || !slices.Equal(out, []string{"text"}) {
			t.Errorf("cfg %+v: modalities = %v / %v", cfg, in, out)
		}
	}
	// Built-in vision needs no projector.
	m.HasBuiltinVision = true
	if !m.HasVision(nil) {
		t.Error("built-in vision not reported")
	}
}

// The cached answer is replaced when the file changes.
func TestProjectorCacheFollowsTheFile(t *testing.T) {
	path := writeProjector(t, yes(), nil, nil)
	if c := ReadProjectorCaps(path); !c.Vision || c.Audio {
		t.Fatalf("first read: %+v", c)
	}
	replacement := writeProjector(t, nil, yes(), yes())
	data, err := os.ReadFile(replacement)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	later := time.Now().Add(time.Minute)
	os.Chtimes(path, later, later)
	if c := ReadProjectorCaps(path); c.Vision || !c.Audio || !c.GenAudio {
		t.Errorf("after the file changed: %+v", c)
	}
}

// The chat template is recorded, so a text-to-speech model can be told
// from one that also chats.
func TestParseGGUFChatTemplate(t *testing.T) {
	with := writeTestGGUF(t, []testKV{
		{"general.architecture", ggufTypeString, "lfm2"},
		{"tokenizer.chat_template", ggufTypeString, "{{ messages }}"},
	})
	without := writeTestGGUF(t, []testKV{{"general.architecture", ggufTypeString, "qwen3tts"}})
	for path, want := range map[string]bool{with: true, without: false} {
		meta, err := ParseGGUFMeta(path)
		if err != nil {
			t.Fatal(err)
		}
		var m Model
		meta.ApplyTo(&m)
		if m.HasChatTemplate != want {
			t.Errorf("%s: HasChatTemplate = %v, want %v", meta.Architecture, m.HasChatTemplate, want)
		}
	}
}

// Fish S2 Pro keeps Qwen3's chat template, tools and all, but it only
// speaks, so it is not offered for tool calls. The same template on a
// chat model is.
func TestSpeechOnlyModelCallsNoTools(t *testing.T) {
	m := &Model{ID: "s2", Arch: "qwen3", HasChatTemplate: true, SupportsTools: true}
	s2 := &ModelConfig{MmprojPath: writeSpeechProjector(t, "fish_s2_gen")}
	if m.CallsTools(s2) || slices.Contains(m.Capabilities(s2), "tools") {
		t.Error("a speech-only model is offered for tool calls")
	}
	if !m.CallsTools(nil) || !slices.Contains(m.Capabilities(nil), "tools") {
		t.Error("the chat model with the same template is not offered for tool calls")
	}
}

// A speech generator takes more GPU memory than its projector file; the
// estimate adds the measured difference for the generators it knows, and
// nothing when the projector is off, kept in system memory, or unknown.
func TestSpeechGenScratchInEstimate(t *testing.T) {
	qwen := writeSpeechProjector(t, "qwen3tts_gen")
	cases := []struct {
		name string
		cfg  *ModelConfig
		want float64
	}{
		{"Qwen3-TTS", &ModelConfig{MmprojPath: qwen}, 1.7},
		{"S2 Pro", &ModelConfig{MmprojPath: writeSpeechProjector(t, "fish_s2_gen")}, 1.5},
		{"a generator not measured", &ModelConfig{MmprojPath: writeSpeechProjector(t, "kokoro")}, 0},
		{"a voice chat model", &ModelConfig{MmprojPath: writeSpeechProjector(t, "lfm3a_gen")}, 0},
		{"an image projector", &ModelConfig{MmprojPath: writeProjector(t, yes(), nil, nil)}, 0},
		{"projector off", &ModelConfig{MmprojPath: qwen, MmprojDisabled: true}, 0},
		{"projector in system memory", &ModelConfig{MmprojPath: qwen, MmprojInRAM: true}, 0},
	}
	m := &Model{ID: "m", SizeBytes: 1 << 30}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := SpeechGenScratchGiB(c.cfg); got != c.want {
				t.Errorf("scratch = %v, want %v", got, c.want)
			}
			b := VRAMBreakdownForConfigOn(m, c.cfg, 1)
			if got, want := b.Aux, AuxFilesVRAMGB(c.cfg)+c.want; got != want {
				t.Errorf("Aux = %v, want %v", got, want)
			}
		})
	}
}
