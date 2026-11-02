package api

import (
	"bytes"
	"encoding/json"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

const testSystemPrompt = "Create game assets for the Godot editor."

// A model's saved system prompt goes into each API's own place for a
// system message, and only when the request has none.
func TestInjectSystemPrompt(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{Enabled: true, SystemPrompt: testSystemPrompt, Aliases: []string{"glb"}})

	cases := []struct {
		name, path, body string
		check            func(t *testing.T, got map[string]any)
	}{
		{
			"chat adds a first system message", "/v1/chat/completions",
			`{"model":"glb","messages":[{"role":"user","content":"Create a mana potion."}]}`,
			func(t *testing.T, got map[string]any) {
				msgs := got["messages"].([]any)
				if len(msgs) != 2 {
					t.Fatalf("messages = %v, want 2", msgs)
				}
				first := msgs[0].(map[string]any)
				if first["role"] != "system" || first["content"] != testSystemPrompt {
					t.Errorf("first message = %v, want the saved system prompt", first)
				}
				if msgs[1].(map[string]any)["content"] != "Create a mana potion." {
					t.Errorf("user message changed: %v", msgs[1])
				}
			},
		},
		{
			"anthropic messages sets system", "/v1/messages",
			`{"model":"glb","messages":[{"role":"user","content":"hi"}]}`,
			func(t *testing.T, got map[string]any) {
				if got["system"] != testSystemPrompt {
					t.Errorf("system = %v, want the saved prompt", got["system"])
				}
			},
		},
		{
			"responses sets instructions", "/v1/responses",
			`{"model":"glb","input":"hi"}`,
			func(t *testing.T, got map[string]any) {
				if got["instructions"] != testSystemPrompt {
					t.Errorf("instructions = %v, want the saved prompt", got["instructions"])
				}
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var got map[string]any
			out := s.injectSystemPrompt(c.path, []byte(c.body))
			if err := json.Unmarshal(out, &got); err != nil {
				t.Fatalf("output is not JSON: %v\n%s", err, out)
			}
			c.check(t, got)
		})
	}
}

// A client that sends its own system message keeps it, and requests the
// prompt does not apply to go upstream byte for byte.
func TestInjectSystemPromptLeavesBodyUntouched(t *testing.T) {
	s := newTestServer(t)
	addProxyModel(t, s, models.ModelConfig{Enabled: true, SystemPrompt: testSystemPrompt})

	noPrompt := newTestServer(t)
	addProxyModel(t, noPrompt, models.ModelConfig{Enabled: true})

	m := `"model":"` + proxyModelID + `"`
	cases := []struct {
		name, path, body string
		s                *Server
	}{
		{"chat with a system message", "/v1/chat/completions", `{` + m + `,"messages":[{"role":"system","content":"mine"},{"role":"user","content":"hi"}]}`, s},
		{"chat with a developer message", "/v1/chat/completions", `{` + m + `,"messages":[{"role":"user","content":"hi"},{"role":"developer","content":"mine"}]}`, s},
		{"chat without messages", "/v1/chat/completions", `{` + m + `,"prompt":"hi"}`, s},
		{"anthropic with system", "/v1/messages", `{` + m + `,"system":[{"type":"text","text":"mine"}],"messages":[]}`, s},
		{"responses with instructions", "/v1/responses", `{` + m + `,"instructions":"mine","input":"hi"}`, s},
		{"responses with a system input", "/v1/responses", `{` + m + `,"input":[{"role":"system","content":"mine"},{"role":"user","content":"hi"}]}`, s},
		{"raw completion", "/completion", `{` + m + `,"prompt":"hi"}`, s},
		{"unknown model", "/v1/chat/completions", `{"model":"nobody","messages":[]}`, s},
		{"not JSON", "/v1/chat/completions", `model=` + proxyModelID, s},
		{"model with no prompt", "/v1/chat/completions", `{` + m + `,"messages":[{"role":"user","content":"hi"}]}`, noPrompt},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := c.s.injectSystemPrompt(c.path, []byte(c.body))
			if !bytes.Equal(out, []byte(c.body)) {
				t.Errorf("body changed:\n got %s\nwant %s", out, c.body)
			}
		})
	}
}

// The config form saves the prompt trimmed and shows it again. A form
// post without the field (another part of the UI) keeps the saved prompt.
func TestConfigFormSavesSystemPrompt(t *testing.T) {
	s := newProfileServer(t, "b100")
	path := "/api/models/" + profTestID + "/config"

	_, body := s.doProfileRequest(t, "PUT", path, url.Values{"system_prompt": {"  " + testSystemPrompt + "\n"}})
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.SystemPrompt != testSystemPrompt {
		t.Fatalf("saved prompt = %q, want %q", cfg.SystemPrompt, testSystemPrompt)
	}
	if !strings.Contains(body, ">"+testSystemPrompt+"</textarea>") {
		t.Errorf("form does not show the saved prompt")
	}

	s.doProfileRequest(t, "PUT", path, url.Values{"context_size": {"8192"}})
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.SystemPrompt != testSystemPrompt {
		t.Errorf("prompt after a post without the field = %q, want it kept", cfg.SystemPrompt)
	}

	s.doProfileRequest(t, "PUT", path, url.Values{"system_prompt": {""}})
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.SystemPrompt != "" {
		t.Errorf("prompt after clearing = %q, want empty", cfg.SystemPrompt)
	}
}
