package models

import (
	"strings"
	"testing"
)

func TestContextLayoutFor(t *testing.T) {
	current := Target{Version: 11364}
	old := Target{Version: 10661}
	tests := []struct {
		name string
		cfg  ModelConfig
		t    Target
		want ContextLayout
	}{
		{"blank is llama.cpp's four shared slots", ModelConfig{ContextSize: 32768}, current,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 32768}},
		{"one at a time", ModelConfig{ContextSize: 32768, Parallel: 1}, current,
			ContextLayout{Slots: 1, Pool: 32768, PerConversation: 32768}},
		{"equal shares", ModelConfig{ContextSize: 32768, Parallel: 4}, current,
			ContextLayout{Slots: 4, Pool: 32768, PerConversation: 8192}},
		{"shared, no limit", ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true}, current,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 32768}},
		{"shared, limit", ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 16384}, current,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 16384}},
		{"shared, limit above the pool", ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 65536}, current,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 32768}},
		{"shared, limit, no context size: pool sized to fit", ModelConfig{Parallel: 4, SharedContext: true, ContextPerSlot: 8192}, current,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 8192}},
		{"shared, build without the limit", ModelConfig{ContextSize: 32768, Parallel: 4, SharedContext: true, ContextPerSlot: 16384}, old,
			ContextLayout{Slots: 4, Shared: true, Pool: 32768, PerConversation: 32768}},
		{"no context size uses the trained one", ModelConfig{Parallel: 2}, current,
			ContextLayout{Slots: 2, Pool: 131072, PerConversation: 65536}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cfg.ContextLayoutFor(131072, tt.t); got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

// What reaches llama-server: blank writes nothing (four shared slots),
// 1 now writes parallel = 1, and the shared keys appear only from two
// conversations with Shared Context on. kv-unified-per-slot needs b10662.
func TestSharedContextOptions(t *testing.T) {
	current := Target{Version: 11364}
	tests := []struct {
		name   string
		cfg    ModelConfig
		t      Target
		ini    []string
		absent []string
		flags  string
	}{
		{"blank", ModelConfig{}, current, nil, []string{"parallel", "kv-unified"}, ""},
		{"one at a time", ModelConfig{Parallel: 1}, current, []string{"parallel = 1"}, []string{"kv-unified"}, "--parallel 1"},
		{"one ignores stored sharing", ModelConfig{Parallel: 1, SharedContext: true, ContextPerSlot: 4096}, current,
			[]string{"parallel = 1"}, []string{"kv-unified"}, "--parallel 1"},
		{"equal shares", ModelConfig{Parallel: 4}, current, []string{"parallel = 4"}, []string{"kv-unified"}, "--parallel 4"},
		{"shared", ModelConfig{Parallel: 4, SharedContext: true}, current,
			[]string{"parallel = 4", "kv-unified = true"}, []string{"kv-unified-per-slot"}, "--parallel 4 --kv-unified"},
		{"shared with limit", ModelConfig{Parallel: 4, SharedContext: true, ContextPerSlot: 16384}, current,
			[]string{"kv-unified = true", "kv-unified-per-slot = 16384"}, nil, "--kv-unified --kv-unified-per-slot 16384"},
		{"shared with limit, old build", ModelConfig{Parallel: 4, SharedContext: true, ContextPerSlot: 16384}, Target{Version: 10661},
			[]string{"kv-unified = true"}, []string{"kv-unified-per-slot"}, "--kv-unified"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
			cfg := tt.cfg
			cfg.Enabled = true
			ini := GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": &cfg}, tt.t)
			for _, want := range tt.ini {
				if !strings.Contains(ini, want+"\n") {
					t.Errorf("want %q in preset:\n%s", want, ini)
				}
			}
			for _, key := range tt.absent {
				if strings.Contains(ini, "\n"+key+" =") {
					t.Errorf("did not want %q in preset:\n%s", key, ini)
				}
			}
			flags := cfg.EffectiveFlagsFor(false, tt.t)
			if tt.flags != "" && !strings.Contains(flags, tt.flags) {
				t.Errorf("want %q in flags: %s", tt.flags, flags)
			}
			if tt.flags == "" && strings.Contains(flags, "--parallel") {
				t.Errorf("blank should write no --parallel: %s", flags)
			}
		})
	}
}
