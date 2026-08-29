package config

import "testing"

// The buffer-size report that the VRAM-estimate work depends on
// (plan/ple-vram-findings.md) only appears at verbosity 4 or above, and
// the router passes its environment to every model instance it starts.
// Curating the variable is what makes that reachable without hand-editing
// the free-form block.
func TestLogVerbosityIsCurated(t *testing.T) {
	var opt *RuntimeEnvOption
	for i, o := range RuntimeEnvOptions() {
		if o.Name == "LLAMA_ARG_LOG_VERBOSITY" {
			opt = &RuntimeEnvOptions()[i]
		}
	}
	if opt == nil {
		t.Fatal("LLAMA_ARG_LOG_VERBOSITY is not a curated option")
	}
	// Backend-independent: this is llama.cpp's own logging, not a GPU knob.
	if len(opt.Backends) != 0 {
		t.Errorf("Backends = %v, want empty (applies to every build)", opt.Backends)
	}
	want := map[string]bool{"": true, "3": true, "4": true, "5": true}
	for _, v := range opt.Values {
		if !want[v] {
			t.Errorf("unexpected value %q", v)
		}
		delete(want, v)
	}
	for v := range want {
		t.Errorf("missing value %q — 4 is the level that reports buffer sizes", v)
	}
}

// A curated value must survive validation, or the setting cannot be saved.
func TestLogVerbosityValidates(t *testing.T) {
	set := EnvSet{Curated: map[string]string{"LLAMA_ARG_LOG_VERBOSITY": "4"}}
	if err := set.Validate(); err != nil {
		t.Errorf("verbosity 4 rejected: %v", err)
	}
	var found bool
	for _, p := range set.Pairs() {
		if p == "LLAMA_ARG_LOG_VERBOSITY=4" {
			found = true
		}
	}
	if !found {
		t.Error("verbosity never reached the environment passed to the router")
	}
}
