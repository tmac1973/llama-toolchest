// Package readonly holds the guard that the JSON stores share: a store
// whose file could not be read or parsed, or was written by a newer
// version, keeps what it loaded in memory but refuses to save, so that
// nothing overwrites the file on disk.
package readonly

import (
	"errors"
	"fmt"
	"io/fs"
)

// Error is a refusal from a read-only store. Its text is the
// plain-language reason; errors.Is matches it against the store's own
// sentinel (for example models.ErrRegistryReadOnly).
type Error struct {
	Reason   string
	Sentinel error
}

func (e Error) Error() string        { return e.Reason }
func (e Error) Is(target error) bool { return target == e.Sentinel }

// Check returns nil when reason is "" (the store is writable), else an
// Error with the reason that matches sentinel.
func Check(reason string, sentinel error) error {
	if reason == "" {
		return nil
	}
	return Error{Reason: reason, Sentinel: sentinel}
}

// ReadReason is the reason for a file that exists but could not be read.
// It returns "" for a missing file, which is a normal first start. file is
// the file name ("models.json"), what is what goes unsaved ("Model
// settings").
func ReadReason(file, what string, err error) string {
	if err == nil || errors.Is(err, fs.ErrNotExist) {
		return ""
	}
	return fmt.Sprintf("%s could not be read (%v). "+
		"%s will not be saved until the file can be read, "+
		"so that nothing overwrites it.", file, err, what)
}

// ParseReason is the reason for a file that could not be parsed.
func ParseReason(file, what string, err error) string {
	return fmt.Sprintf("%s could not be parsed (%v). "+
		"%s will not be saved until the file is fixed, "+
		"so that nothing overwrites it.", file, err, what)
}

// NewerReason is the reason for a file written by a newer version, whose
// schema this build does not know.
func NewerReason(file, what string, got, supported int) string {
	return fmt.Sprintf("%s was written by a newer version of "+
		"llama-toolchest (schema %d; this version reads up to %d). "+
		"%s will not be saved until llama-toolchest is upgraded, "+
		"so that nothing the newer version stored is lost.",
		file, got, supported, what)
}
