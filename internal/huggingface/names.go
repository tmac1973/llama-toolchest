package huggingface

import "strings"

// SafeModelID returns the URL/filesystem-safe form of an HF model ID:
// slashes become "--". It names the per-model download directory and
// prefixes download and registry IDs — the single canonical definition
// of the transformation.
func SafeModelID(modelID string) string {
	return strings.ReplaceAll(modelID, "/", "--")
}

// SafeFileID returns the URL-safe ID fragment for a GGUF filename: the
// ".gguf" suffix is dropped and slashes become "--".
func SafeFileID(filename string) string {
	return strings.ReplaceAll(strings.TrimSuffix(filename, ".gguf"), "/", "--")
}
