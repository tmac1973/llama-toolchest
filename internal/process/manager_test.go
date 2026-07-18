package process

import (
	"strings"
	"testing"
)

// countCUDAOrder returns how many CUDA_DEVICE_ORDER entries are in env and the
// value of the last one (the one that wins).
func countCUDAOrder(env []string) (n int, last string) {
	for _, kv := range env {
		if strings.HasPrefix(kv, "CUDA_DEVICE_ORDER=") {
			n++
			last = strings.TrimPrefix(kv, "CUDA_DEVICE_ORDER=")
		}
	}
	return n, last
}

// TestPinCUDADeviceOrderAdds pins issue #68: when the user hasn't set
// CUDA_DEVICE_ORDER, we add PCI_BUS_ID so the CUDA backend's device indices
// match nvidia-smi (and thus the web UI's "GPU N" labels).
func TestPinCUDADeviceOrderAdds(t *testing.T) {
	env := pinCUDADeviceOrder([]string{"PATH=/usr/bin", "HOME=/root"})
	n, last := countCUDAOrder(env)
	if n != 1 || last != "PCI_BUS_ID" {
		t.Fatalf("expected exactly one CUDA_DEVICE_ORDER=PCI_BUS_ID; got count=%d last=%q env=%v", n, last, env)
	}
}

// TestPinCUDADeviceOrderRespectsExisting verifies a user-provided value is
// left untouched — we never override an explicit choice.
func TestPinCUDADeviceOrderRespectsExisting(t *testing.T) {
	env := pinCUDADeviceOrder([]string{"PATH=/usr/bin", "CUDA_DEVICE_ORDER=FASTEST_FIRST"})
	n, last := countCUDAOrder(env)
	if n != 1 || last != "FASTEST_FIRST" {
		t.Fatalf("expected the existing FASTEST_FIRST to survive untouched; got count=%d last=%q env=%v", n, last, env)
	}
}

// applyExtraEnv must not override a variable the operator exported
// around the process — a systemd drop-in or container env stays
// authoritative over a value set in the UI.
func TestApplyExtraEnvDoesNotOverrideInherited(t *testing.T) {
	env := []string{"PATH=/usr/bin", "ROCBLAS_USE_HIPBLASLT=0"}
	got := applyExtraEnv(env, []string{"ROCBLAS_USE_HIPBLASLT=1", "GGML_CUDA_DISABLE_GRAPHS=1"})

	var hipblaslt, graphs string
	for _, kv := range got {
		if v, ok := strings.CutPrefix(kv, "ROCBLAS_USE_HIPBLASLT="); ok {
			hipblaslt = v
		}
		if v, ok := strings.CutPrefix(kv, "GGML_CUDA_DISABLE_GRAPHS="); ok {
			graphs = v
		}
	}
	if hipblaslt != "0" {
		t.Errorf("inherited ROCBLAS_USE_HIPBLASLT = %q, want the inherited 0 to win", hipblaslt)
	}
	if graphs != "1" {
		t.Errorf("GGML_CUDA_DISABLE_GRAPHS = %q, want 1", graphs)
	}
}

func TestApplyExtraEnvNoExtras(t *testing.T) {
	env := []string{"PATH=/usr/bin"}
	if got := applyExtraEnv(env, nil); len(got) != 1 {
		t.Errorf("got %v, want the environment unchanged", got)
	}
}
