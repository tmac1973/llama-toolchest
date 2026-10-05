// Package atomicfile writes files so that a reader, or the next start
// after a crash, sees either the old contents or the new ones — never a
// truncated mix.
package atomicfile

import (
	"os"
	"path/filepath"
)

// Write writes data to a temporary file beside path, syncs it, and renames
// it over path. The directory is created if it does not exist, and the
// file ends up with mode 0644.
func Write(path string, data []byte) error {
	return WriteMode(path, data, 0o644)
}

// WriteMode is Write with the file's mode given, for files that hold
// secrets (0600).
func WriteMode(path string, data []byte, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op once the rename has happened
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), perm); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}
