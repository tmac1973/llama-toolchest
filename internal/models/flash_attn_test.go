package models

import (
	"strings"
	"testing"
)

func renderFA(cfg *ModelConfig) string {
	m := &Model{ID: "m", Filename: "m.gguf", FilePath: "/models/m.gguf"}
	return GeneratePresetINI("/models", []*Model{m}, map[string]*ModelConfig{"m": cfg}, "rocm")
}

// Writing nothing for "off" left llama.cpp on its own default, which is
// auto — flash attention enabled wherever the backend supports it. So the
// toggle did nothing in the off direction while every surface in the app
// reported it as off.
//
// Confirmed on hardware: a model saved with flash attention off produced
// byte-identical compute buffers (378.02 MiB per card) and identical total
// VRAM (48.09 GiB) to the same model with it on. Passing the flag
// explicitly changed the load outcome, proving the value does reach
// llama.cpp once it is actually written.
func TestFlashAttentionEmittedInBothDirections(t *testing.T) {
	for _, tt := range []struct {
		on   bool
		want string
	}{
		{true, "flash-attn = on"},
		{false, "flash-attn = off"},
	} {
		ini := renderFA(&ModelConfig{Enabled: true, ContextSize: 4096, FlashAttention: tt.on})
		if !strings.Contains(ini, tt.want) {
			t.Errorf("flash attention %v did not emit %q:\n%s", tt.on, tt.want, ini)
		}
	}
}

// The off case is the regression: assert the opposite value is absent, so
// a future edit cannot satisfy the test above by emitting both.
func TestFlashAttentionOffDoesNotEmitOn(t *testing.T) {
	ini := renderFA(&ModelConfig{Enabled: true, ContextSize: 4096, FlashAttention: false})
	if strings.Contains(ini, "flash-attn = on") {
		t.Errorf("off config emitted the on value:\n%s", ini)
	}
}

// llama.cpp fails a tensor-split load without flash attention, after
// reading the weights and with nothing tying the error to the toggle.
// Rejecting the save is the only place a user can act on it.
func TestTensorSplitRequiresFlashAttention(t *testing.T) {
	bad := &ModelConfig{FlashAttention: false, SplitMode: "tensor"}
	if err := bad.ValidateFlashAttention(); err == nil {
		t.Error("accepted a combination llama.cpp refuses to load")
	} else if !strings.Contains(err.Error(), "layer split") {
		t.Errorf("error does not say what to do instead: %v", err)
	}
	for _, ok := range []*ModelConfig{
		{FlashAttention: true, SplitMode: "tensor"},
		{FlashAttention: false, SplitMode: "layer"},
		{FlashAttention: false, SplitMode: ""},
	} {
		if err := ok.ValidateFlashAttention(); err != nil {
			t.Errorf("rejected a valid pair (fa=%v split=%q): %v", ok.FlashAttention, ok.SplitMode, err)
		}
	}
}

// llama.cpp's second requirement, found when an A/B of flash attention
// failed to load at all: "quantized V cache requires flash_attn to be
// enabled". More likely to be met by accident than the tensor-split rule,
// since a quantized cache is a common way to save memory.
func TestQuantizedKVCacheRequiresFlashAttention(t *testing.T) {
	for _, quant := range []string{"q8_0", "q4_0"} {
		bad := &ModelConfig{FlashAttention: false, KVCacheQuant: quant}
		if err := bad.ValidateFlashAttention(); err == nil {
			t.Errorf("accepted flash attention off with a %s cache", quant)
		} else if !strings.Contains(err.Error(), quant) {
			t.Errorf("error does not name the cache type: %v", err)
		}
	}
	for _, ok := range []*ModelConfig{
		{FlashAttention: true, KVCacheQuant: "q8_0"},
		{FlashAttention: false, KVCacheQuant: ""},
	} {
		if err := ok.ValidateFlashAttention(); err != nil {
			t.Errorf("rejected a valid pair (fa=%v kv=%q): %v", ok.FlashAttention, ok.KVCacheQuant, err)
		}
	}
}
