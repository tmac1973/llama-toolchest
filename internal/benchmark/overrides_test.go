package benchmark

import (
	"reflect"
	"testing"
)

// ConfigOverrides drifted away from ConfigSnapshot once already: six
// fields were accepted by the job form, persisted, and then silently
// dropped because nothing downstream could carry them. Every override
// field must have a destination — either the router config
// (ConfigSnapshot, applied via the preset INI) or the per-request
// sampling params (SamplingParams, sent with the completion).
//
// If this fails you added a field to ConfigOverrides without wiring it
// anywhere. Add it to whichever destination applies, and to
// applyOverrides or samplingFromOverrides.
func TestEveryOverrideFieldHasADestination(t *testing.T) {
	destinations := map[string]bool{}
	for _, typ := range []reflect.Type{
		reflect.TypeOf(ConfigSnapshot{}),
		reflect.TypeOf(SamplingParams{}),
	} {
		for i := 0; i < typ.NumField(); i++ {
			destinations[typ.Field(i).Name] = true
		}
	}

	overrides := reflect.TypeOf(ConfigOverrides{})
	for i := 0; i < overrides.NumField(); i++ {
		name := overrides.Field(i).Name
		if !destinations[name] {
			t.Errorf("ConfigOverrides.%s has no destination in ConfigSnapshot or SamplingParams — it would be silently dropped", name)
		}
	}
}

func TestApplyOverridesCoversEveryConfigSnapshotField(t *testing.T) {
	// Build an overrides struct with every pointer field populated, then
	// assert applyOverrides moved each one onto the snapshot. Catches a
	// field added to both structs but forgotten in applyOverrides.
	ngl, ctx, threads := 42, 4242, 24
	fa, dio := true, true
	kv, ga, ts, st, dmp := "q4_0", "0", "1,2", "draft", "/models/draft.gguf"

	got := applyOverrides(ConfigSnapshot{}, &ConfigOverrides{
		GPULayers:      &ngl,
		ContextSize:    &ctx,
		Threads:        &threads,
		FlashAttention: &fa,
		DirectIO:       &dio,
		KVCacheQuant:   &kv,
		GPUAssign:      &ga,
		TensorSplit:    &ts,
		SpecType:       &st,
		DraftModelPath: &dmp,
	})

	want := ConfigSnapshot{
		GPULayers: ngl, ContextSize: ctx, Threads: threads,
		FlashAttention: fa, DirectIO: dio, KVCacheQuant: kv,
		GPUAssign: ga, TensorSplit: ts, SpecType: st, DraftModelPath: dmp,
	}
	if got != want {
		t.Errorf("applyOverrides dropped a field:\ngot  %+v\nwant %+v", got, want)
	}
}

func TestApplyOverridesNilLeavesBaseUntouched(t *testing.T) {
	base := ConfigSnapshot{GPULayers: 999, ContextSize: 8192, Threads: 8}
	if got := applyOverrides(base, nil); got != base {
		t.Errorf("nil overrides mutated base: got %+v want %+v", got, base)
	}
}

func TestSamplingFromOverridesNil(t *testing.T) {
	if got := samplingFromOverrides(nil); got != (SamplingParams{}) {
		t.Errorf("nil overrides = %+v, want zero", got)
	}
}

func TestSamplingAppliedToRequestBody(t *testing.T) {
	temp, topP, minP, rep := 0.7, 0.9, 0.05, 1.1
	topK := 40
	s := samplingFromOverrides(&ConfigOverrides{
		Temperature: &temp, TopP: &topP, TopK: &topK, MinP: &minP, RepeatPenalty: &rep,
	})

	body := map[string]any{"model": "m"}
	s.applyTo(body)

	for key, want := range map[string]any{
		"temperature":    0.7,
		"top_p":          0.9,
		"top_k":          40,
		"min_p":          0.05,
		"repeat_penalty": 1.1,
	} {
		if body[key] != want {
			t.Errorf("body[%q] = %v, want %v", key, body[key], want)
		}
	}
}

// Unset sampling fields must not appear in the request at all, so
// llama-server keeps its own defaults rather than receiving a zero.
func TestSamplingOmitsUnsetFields(t *testing.T) {
	temp := 0.7
	s := SamplingParams{Temperature: &temp}

	body := map[string]any{}
	s.applyTo(body)

	if _, ok := body["temperature"]; !ok {
		t.Error("temperature should be present")
	}
	for _, key := range []string{"top_p", "top_k", "min_p", "repeat_penalty"} {
		if v, ok := body[key]; ok {
			t.Errorf("unset %s was sent as %v; it must be omitted", key, v)
		}
	}
}
