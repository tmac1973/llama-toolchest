package atomicfile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteReplacesAndLeavesNoTempFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "data.json")

	if err := Write(path, []byte("first")); err != nil {
		t.Fatal(err)
	}
	if err := Write(path, []byte("second")); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "second" {
		t.Fatalf("contents = %q, want %q", got, "second")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o644 {
		t.Errorf("mode = %v, want 0644", info.Mode().Perm())
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("directory holds %d entries, want only data.json", len(entries))
	}
}

// A copy cut short leaves no file at the target that looks finished.
func TestCopyLeavesNoPartOnError(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.gguf")
	if err := Copy(filepath.Join(dir, "missing.gguf"), dst); err == nil {
		t.Fatal("copying a missing file succeeded")
	}
	for _, p := range []string{dst, dst + ".part"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s exists after a failed copy", p)
		}
	}
}

// Copy writes the source's bytes to the target and leaves no .part file.
func TestCopyCopiesAndRemovesPart(t *testing.T) {
	dir := t.TempDir()
	src, dst := filepath.Join(dir, "in"), filepath.Join(dir, "out")
	if err := os.WriteFile(src, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := Copy(src, dst); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(dst); err != nil || string(got) != "data" {
		t.Errorf("target = %q (err %v), want data", got, err)
	}
	if _, err := os.Stat(dst + ".part"); !os.IsNotExist(err) {
		t.Errorf(".part file left behind")
	}
}
