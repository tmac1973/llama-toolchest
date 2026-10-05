// Package libpath sets the library search path for a child process, so a
// llama.cpp build binary finds the shared libraries next to it
// (libllama.so, libggml*.so, ...).
package libpath

import (
	"os"
	"runtime"
	"strings"
)

// Prepend returns env with dir put first in the loader's search path:
// LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS, PATH on Windows
// (where DLLs are found through PATH). An existing value is kept after
// dir, so libraries the environment already points at (the CUDA or ROCm
// runtime in the container) are still found. env is modified in place
// when the variable exists.
func Prepend(env []string, dir string) []string {
	return prependVar(env, varName(runtime.GOOS), dir)
}

func varName(goos string) string {
	switch goos {
	case "darwin":
		return "DYLD_LIBRARY_PATH"
	case "windows":
		return "PATH"
	}
	return "LD_LIBRARY_PATH"
}

func prependVar(env []string, name, dir string) []string {
	prefix := name + "="
	for i, kv := range env {
		// Windows environment names are not case sensitive ("Path").
		if len(kv) >= len(prefix) && strings.EqualFold(kv[:len(prefix)], prefix) {
			if old := kv[len(prefix):]; old != "" {
				env[i] = kv[:len(prefix)] + dir + string(os.PathListSeparator) + old
			} else {
				env[i] = kv[:len(prefix)] + dir
			}
			return env
		}
	}
	return append(env, prefix+dir)
}
