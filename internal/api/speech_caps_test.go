package api

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// writeSpeechProjector writes a projector file that reads and makes
// audio, like Qwen3-TTS's and LFM3-Audio's.
func writeSpeechProjector(t *testing.T) string {
	t.Helper()
	var buf bytes.Buffer
	str := func(s string) {
		binary.Write(&buf, binary.LittleEndian, uint64(len(s)))
		buf.WriteString(s)
	}
	buf.WriteString("GGUF")
	binary.Write(&buf, binary.LittleEndian, uint32(3))
	binary.Write(&buf, binary.LittleEndian, uint64(0))
	binary.Write(&buf, binary.LittleEndian, uint64(3))
	str("general.architecture")
	binary.Write(&buf, binary.LittleEndian, uint32(8)) // string
	str("clip")
	for _, key := range []string{"clip.has_audio_encoder", "clip.has_gen_audio_encoder"} {
		str(key)
		binary.Write(&buf, binary.LittleEndian, uint32(7)) // bool
		buf.WriteByte(1)
	}
	path := filepath.Join(t.TempDir(), "mmproj.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// /v1/models and the capabilities object report a text-to-speech model
// as one that speaks and does not chat, with llama-server's architecture
// lists, and a voice chat model as one that does all three.
func TestSpeechModelCapabilities(t *testing.T) {
	for _, c := range []struct {
		name         string
		chatTemplate bool
		caps, out    []string
	}{
		{"text-to-speech only", false, []string{"speech"}, []string{"audio"}},
		{"voice chat", true, []string{"chat", "audio", "speech"}, []string{"text", "audio"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			s := newTestServer(t)
			m := addProxyModel(t, s, models.ModelConfig{Enabled: true, MmprojPath: writeSpeechProjector(t)})
			m.HasChatTemplate = c.chatTemplate

			rec := httptest.NewRecorder()
			s.handleV1Models(rec, httptest.NewRequest("GET", "/v1/models", nil))
			var list struct {
				Data []struct {
					Meta struct {
						Capabilities []string `json:"capabilities"`
						Architecture struct {
							In  []string `json:"input_modalities"`
							Out []string `json:"output_modalities"`
						} `json:"architecture"`
					} `json:"meta"`
				} `json:"data"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil || len(list.Data) != 1 {
				t.Fatalf("answer: %v\n%s", err, rec.Body)
			}
			meta := list.Data[0].Meta
			if !slices.Equal(meta.Capabilities, c.caps) {
				t.Errorf("capabilities = %v, want %v", meta.Capabilities, c.caps)
			}
			if !slices.Equal(meta.Architecture.In, []string{"text", "audio"}) || !slices.Equal(meta.Architecture.Out, c.out) {
				t.Errorf("architecture = %v / %v, want [text audio] / %v", meta.Architecture.In, meta.Architecture.Out, c.out)
			}

			cfg, _ := s.registry.GetConfig(m.ID)
			obj := s.buildCapabilities(m, cfg)
			if obj["speech"] != true || obj["chat"] != c.chatTemplate || obj["audio"] != c.chatTemplate || obj["vision"] != false {
				t.Errorf("capabilities object: chat %v audio %v speech %v vision %v", obj["chat"], obj["audio"], obj["speech"], obj["vision"])
			}
		})
	}
}
