// Package jsonstore reads and writes the versioned JSON files of the
// generation and fork stores: one object with a "schema_version" field,
// written with mode 0600 because instances can carry environment values.
package jsonstore

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmac1973/llama-toolchest/internal/atomicfile"
)

// Load reads the file at path into v. The file's schema_version must be
// version. A missing file is not an error: v is left as it is, which is
// the empty store.
func Load(path string, version int, v any) error {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	var head struct {
		SchemaVersion int `json:"schema_version"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	if head.SchemaVersion != version {
		return fmt.Errorf("%s has schema version %d; this build reads version %d",
			path, head.SchemaVersion, version)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		return fmt.Errorf("reading %s: %w", path, err)
	}
	return nil
}

// Save writes v to path, creating its folder. what names the store in
// errors ("generation store").
func Save(path, what string, v any) error {
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Errorf("saving %s: %w", what, err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("saving %s: %w", what, err)
	}
	if err := atomicfile.WriteMode(path, raw, 0o600); err != nil {
		return fmt.Errorf("saving %s to %s: %w", what, path, err)
	}
	return nil
}
