package builder

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// writeConfigFile puts a file into <dir>/config and returns its path.
func writeConfigFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, "config", name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// A builds.json that cannot be parsed is left as it is: the build history
// goes read-only instead of being saved over with an empty list.
func TestCorruptBuildsFileIsNotOverwritten(t *testing.T) {
	dir := t.TempDir()
	const corrupt = `[{"id": "b1", "status": `
	p := writeConfigFile(t, dir, "builds.json", corrupt)

	b := NewBuilder(dir)
	if reasons := b.ReadOnlyReasons(); len(reasons) != 1 {
		t.Fatalf("ReadOnlyReasons = %q, want one reason for builds.json", reasons)
	}
	b.saveBuilds()
	if got, _ := os.ReadFile(p); string(got) != corrupt {
		t.Errorf("builds.json was rewritten: %q", got)
	}
}

// The same for the saved flag sets: a save or a delete is refused with the
// reason, and the file on disk stays as it was.
func TestCorruptFlagPresetsFileIsNotOverwritten(t *testing.T) {
	dir := t.TempDir()
	const corrupt = `[{"name": "fast", `
	p := writeConfigFile(t, dir, "build-flag-presets.json", corrupt)

	b := NewBuilder(dir)
	err := b.SaveFlagPreset(FlagPreset{Name: "fast", Profile: "cuda", Options: map[string]bool{}})
	if !errors.Is(err, ErrFlagPresetsReadOnly) {
		t.Fatalf("SaveFlagPreset error = %v, want ErrFlagPresetsReadOnly", err)
	}
	if _, err := b.DeleteFlagPreset("fast"); !errors.Is(err, ErrFlagPresetsReadOnly) {
		t.Fatalf("DeleteFlagPreset error = %v, want ErrFlagPresetsReadOnly", err)
	}
	if got, _ := os.ReadFile(p); string(got) != corrupt {
		t.Errorf("build-flag-presets.json was rewritten: %q", got)
	}
	if reasons := b.ReadOnlyReasons(); len(reasons) != 1 {
		t.Errorf("ReadOnlyReasons = %q, want one reason", reasons)
	}
}

// A missing file is a normal first start, not a reason to go read-only.
func TestMissingBuilderFilesAreWritable(t *testing.T) {
	b := NewBuilder(t.TempDir())
	if reasons := b.ReadOnlyReasons(); len(reasons) != 0 {
		t.Errorf("ReadOnlyReasons = %q, want none", reasons)
	}
}
