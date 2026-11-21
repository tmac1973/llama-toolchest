package builder

import "testing"

func TestIsSharedLib(t *testing.T) {
	for name, want := range map[string]bool{
		"libllama.so":                true,
		"libggml.so.0":               true,
		"libllama-server-impl.dylib": true,
		"libggml-metal.1.0.0.dylib":  true,
		"llama-server":               false,
		"libggml.a":                  false,
		"notes.solution":             false,
	} {
		if got := isSharedLib(name); got != want {
			t.Errorf("isSharedLib(%q) = %v, want %v", name, got, want)
		}
	}
}
