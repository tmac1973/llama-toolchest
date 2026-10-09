package agentconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// marshal writes indented JSON with a trailing newline, without escaping
// <, > and & (URLs and model names read better unescaped).
func marshal(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// keyOr returns the agent's reference to KeyEnv when the server needs a key,
// else the placeholder agents accept for "no key".
func keyOr(in Input, ref string) string {
	if in.APIKey {
		return ref
	}
	return "none"
}

// ── opencode and Kilo Code ──────────────────────────────────────────────────
//
// Kilo Code is built on opencode and reads the same schema. Both deep-merge
// every config layer, so the file can be handed over whole and layered on
// with OPENCODE_CONFIG (opencode) or saved as the user config (Kilo).
// Docs: https://opencode.ai/docs/providers/ and https://opencode.ai/docs/config/

type ocModel struct {
	Name       string       `json:"name"`
	ToolCall   bool         `json:"tool_call"`
	Reasoning  bool         `json:"reasoning"`
	Attachment bool         `json:"attachment"`
	Modalities ocModalities `json:"modalities"`
	Limit      ocLimit      `json:"limit"`
}

type ocModalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

// ocLimit must be set: unset limits read as 0 and break compaction.
type ocLimit struct {
	Context int `json:"context"`
	Output  int `json:"output"`
}

func openCodeConfig(in Input, schema string) ([]byte, error) {
	models := map[string]ocModel{}
	for _, m := range in.Models {
		input := []string{"text"}
		if m.Vision {
			input = append(input, "image")
		}
		models[m.ID] = ocModel{
			Name:       m.ID,
			ToolCall:   m.Tools,
			Reasoning:  m.Reasoning,
			Attachment: m.Vision,
			Modalities: ocModalities{Input: input, Output: []string{"text"}},
			Limit:      ocLimit{Context: m.Context, Output: m.MaxOutput},
		}
	}
	cfg := map[string]any{
		"provider": map[string]any{
			ProviderID: map[string]any{
				// openai-compatible talks /v1/chat/completions; the plain
				// openai package would use /v1/responses.
				"npm":  "@ai-sdk/openai-compatible",
				"name": ProviderName,
				"options": map[string]any{
					"baseURL": in.BaseURL,
					"apiKey":  keyOr(in, "{env:"+KeyEnv+"}"),
				},
				"models": models,
			},
		},
	}
	if schema != "" {
		cfg["$schema"] = schema
	}
	if len(in.Models) > 0 {
		def := ProviderID + "/" + in.Models[0].ID
		cfg["model"] = def
		cfg["small_model"] = def
	}
	return marshal(cfg)
}

var opencode = Agent{
	ID:       "opencode",
	Name:     "opencode",
	Homepage: "https://opencode.ai/docs/providers/",
	FileName: "llama-toolchest.opencode.json",
	SavePath: map[OS]string{
		Linux:   "~/.config/opencode/",
		MacOS:   "~/.config/opencode/",
		Windows: `%USERPROFILE%\.config\opencode\`,
	},
	Activate: "export OPENCODE_CONFIG=~/.config/opencode/llama-toolchest.opencode.json",
	Notes: []string{
		"OPENCODE_CONFIG layers this file over your own opencode.json, so nothing of yours is replaced. Add the export to your shell profile to make it stick.",
	},
	Generate: func(in Input) ([]byte, error) {
		return openCodeConfig(in, "https://opencode.ai/config.json")
	},
}

var kilo = Agent{
	ID:       "kilo",
	Name:     "Kilo Code",
	Homepage: "https://kilo.ai/docs",
	FileName: "kilo.json",
	SavePath: map[OS]string{
		Linux:   "~/.config/kilo/",
		MacOS:   "~/.config/kilo/",
		Windows: `%USERPROFILE%\.config\kilo\`,
	},
	Notes: []string{
		"Kilo merges this with a kilo.json in your project, if you have one. If you already have a ~/.config/kilo/kilo.json, copy the \"provider\" block into it instead.",
	},
	Generate: func(in Input) ([]byte, error) {
		return openCodeConfig(in, "")
	},
}

// ── pi ──────────────────────────────────────────────────────────────────────
//
// pi reads custom providers from ~/.pi/agent/models.json, one file, so a
// user who already has one merges the provider in. A model stays hidden in
// /model until its key resolves, hence the placeholder.
// Docs: packages/coding-agent/docs/models.md in earendil-works/pi.

type piModel struct {
	ID            string   `json:"id"`
	Name          string   `json:"name"`
	Reasoning     bool     `json:"reasoning"`
	Input         []string `json:"input"`
	ContextWindow int      `json:"contextWindow"`
	MaxTokens     int      `json:"maxTokens"`
	Cost          piCost   `json:"cost"`
}

type piCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

var pi = Agent{
	ID:       "pi",
	Name:     "pi",
	Homepage: "https://github.com/earendil-works/pi",
	FileName: "models.json",
	SavePath: map[OS]string{
		Linux:   "~/.pi/agent/",
		MacOS:   "~/.pi/agent/",
		Windows: `%USERPROFILE%\.pi\agent\`,
	},
	Notes: []string{
		"If you already have a models.json, copy the \"llama-toolchest\" entry into its \"providers\" instead of replacing the file.",
		"Pick the model with /model in pi.",
	},
	Generate: func(in Input) ([]byte, error) {
		var ms []piModel
		for _, m := range in.Models {
			input := []string{"text"}
			if m.Vision {
				input = append(input, "image")
			}
			ms = append(ms, piModel{
				ID: m.ID, Name: m.ID, Reasoning: m.Reasoning, Input: input,
				ContextWindow: m.Context, MaxTokens: m.MaxOutput,
			})
		}
		return marshal(map[string]any{
			"providers": map[string]any{
				ProviderID: map[string]any{
					"baseUrl": in.BaseURL,
					"api":     "openai-completions",
					"apiKey":  keyOr(in, "$"+KeyEnv),
					"models":  ms,
				},
			},
		})
	},
}

// ── Goose ───────────────────────────────────────────────────────────────────
//
// Goose reads each file in custom_providers/ as one provider, so this one
// sits beside the user's config without touching it. base_url is the full
// chat completions URL, not the /v1 root.
// Docs: documentation/docs/getting-started/providers.md in aaif-goose/goose.

type gooseModel struct {
	Name         string `json:"name"`
	ContextLimit int    `json:"context_limit"`
}

// gooseID is the provider name; Goose wants an identifier without hyphens.
const gooseID = "llama_toolchest"

var goose = Agent{
	ID:       "goose",
	Name:     "Goose",
	Homepage: "https://block.github.io/goose/docs/getting-started/providers",
	FileName: gooseID + ".json",
	SavePath: map[OS]string{
		Linux:   "~/.config/goose/custom_providers/",
		MacOS:   "~/.config/goose/custom_providers/",
		Windows: `%APPDATA%\Block\goose\config\custom_providers\`,
	},
	Activate: "goose session --provider " + gooseID,
	Notes: []string{
		"Or choose llama-toolchest in goose configure to make it the default.",
		"Goose relies on tool calling: use a model that supports tools.",
	},
	Generate: func(in Input) ([]byte, error) {
		var ms []gooseModel
		for _, m := range in.Models {
			ms = append(ms, gooseModel{Name: m.ID, ContextLimit: m.Context})
		}
		cfg := map[string]any{
			"name":               gooseID,
			"engine":             "openai",
			"display_name":       ProviderName,
			"description":        "Local models served by llama-toolchest",
			"base_url":           strings.TrimSuffix(in.BaseURL, "/") + "/chat/completions",
			"models":             ms,
			"supports_streaming": true,
			"requires_auth":      in.APIKey,
		}
		if in.APIKey {
			cfg["api_key_env"] = KeyEnv
		}
		return marshal(cfg)
	},
}

// ── Crush ───────────────────────────────────────────────────────────────────
//
// Crush's config is a script of commands (crushrc), merged across the
// global and project files, so this one is sourced from the user's own
// rather than replacing it. There is no tool flag: Crush assumes tools.
// Docs: docs/config/README.md in charmbracelet/crush.

var crush = Agent{
	ID:       "crush",
	Name:     "Crush",
	Homepage: "https://github.com/charmbracelet/crush",
	FileName: "llama-toolchest.crushrc",
	SavePath: map[OS]string{
		Linux:   "~/.config/crush/",
		MacOS:   "~/.config/crush/",
		Windows: `%USERPROFILE%\.config\crush\`,
	},
	Activate: "echo 'source ~/.config/crush/llama-toolchest.crushrc' >> ~/.config/crush/crushrc",
	Notes: []string{
		"The source line keeps your own crushrc as it is; re-downloading this file updates the models without editing it again.",
	},
	Generate: func(in Input) ([]byte, error) {
		var b strings.Builder
		b.WriteString(header(in, "#"))
		b.WriteString("\n")
		fmt.Fprintf(&b, "provider add %s --name %q --type openai-compat \\\n", ProviderID, ProviderName)
		fmt.Fprintf(&b, "  --base-url %q --api-key %q\n", in.BaseURL, keyOr(in, "${"+KeyEnv+"}"))
		for _, m := range in.Models {
			fmt.Fprintf(&b, "model add %s/%s --name %q \\\n", ProviderID, m.ID, m.ID)
			fmt.Fprintf(&b, "  --context-window %d --default-max-tokens %d --can-reason %t --supports-images %t\n",
				m.Context, m.MaxOutput, m.Reasoning, m.Vision)
		}
		if len(in.Models) > 0 {
			def := ProviderID + "/" + in.Models[0].ID
			fmt.Fprintf(&b, "model large %s\nmodel small %s\n", def, def)
		}
		// Loading a model on first use can outlast the 60s default.
		b.WriteString("option request-timeout 600\n")
		return []byte(b.String()), nil
	},
}
