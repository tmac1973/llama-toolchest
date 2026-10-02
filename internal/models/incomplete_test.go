package models

import (
	"os"
	"path/filepath"
	"testing"
)

// touch creates a file with the given size (bytes of zero content) under dir,
// making any parent directories as needed.
func touch(t *testing.T, path string, size int) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
}

// register adds a model whose primary file is filename inside modelsDir/<safeName>.
func register(t *testing.T, r *Registry, modelsDir, modelID, filename string) {
	t.Helper()
	safe := filepath.Join(modelsDir, sanitizeDir(modelID))
	m := &Model{
		ID:       modelID + "--" + filename,
		ModelID:  modelID,
		Filename: filename,
		FilePath: filepath.Join(safe, filename),
	}
	if err := r.Add(m); err != nil {
		t.Fatal(err)
	}
}

func sanitizeDir(modelID string) string {
	out := ""
	for _, c := range modelID {
		if c == '/' {
			out += "--"
		} else {
			out += string(c)
		}
	}
	return out
}

func TestIncompleteRegistered(t *testing.T) {
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models")
	r := NewRegistry(dir, modelsDir)

	// (a) registered sharded model missing shard 2 of 3 → incomplete.
	aDir := filepath.Join(modelsDir, "org--repoA")
	touch(t, filepath.Join(aDir, "modelA-00001-of-00003.gguf"), 10)
	touch(t, filepath.Join(aDir, "modelA-00003-of-00003.gguf"), 10)
	register(t, r, modelsDir, "org/repoA", "modelA-00001-of-00003.gguf")

	// (c) registered sharded model with all shards present → complete.
	cDir := filepath.Join(modelsDir, "org--repoC")
	touch(t, filepath.Join(cDir, "modelC-00001-of-00002.gguf"), 10)
	touch(t, filepath.Join(cDir, "modelC-00002-of-00002.gguf"), 10)
	register(t, r, modelsDir, "org/repoC", "modelC-00001-of-00002.gguf")

	// (d) registered single-file model, present → complete.
	dDir := filepath.Join(modelsDir, "org--repoD")
	touch(t, filepath.Join(dDir, "modelD.gguf"), 10)
	register(t, r, modelsDir, "org/repoD", "modelD.gguf")

	got := r.IncompleteRegistered()
	if len(got) != 1 {
		t.Fatalf("IncompleteRegistered returned %d models, want 1: %+v", len(got), got)
	}
	if got[0].ModelID != "org/repoA" {
		t.Errorf("incomplete model = %q, want org/repoA", got[0].ModelID)
	}
}

func TestOrphanParts(t *testing.T) {
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models")
	r := NewRegistry(dir, modelsDir)

	// (b) unregistered partial: two .part shards, no registry entry.
	bDir := filepath.Join(modelsDir, "org--repoB")
	touch(t, filepath.Join(bDir, "modelB-00001-of-00002.gguf.part"), 100)
	touch(t, filepath.Join(bDir, "modelB-00002-of-00002.gguf.part"), 50)

	// A registered-but-incomplete model whose shard 2 is a .part — must NOT
	// appear in OrphanParts (it surfaces via IncompleteRegistered instead).
	eDir := filepath.Join(modelsDir, "org--repoE")
	touch(t, filepath.Join(eDir, "modelE-00001-of-00002.gguf"), 10)
	touch(t, filepath.Join(eDir, "modelE-00002-of-00002.gguf.part"), 5)
	register(t, r, modelsDir, "org/repoE", "modelE-00001-of-00002.gguf")

	got := r.OrphanParts()
	if len(got) != 1 {
		t.Fatalf("OrphanParts returned %d entries, want 1: %+v", len(got), got)
	}
	p := got[0]
	if p.ModelID != "org/repoB" {
		t.Errorf("orphan ModelID = %q, want org/repoB", p.ModelID)
	}
	if p.Filename != "modelB-00001-of-00002.gguf" {
		t.Errorf("orphan resume Filename = %q, want first shard", p.Filename)
	}
	if p.PartCount != 2 {
		t.Errorf("orphan PartCount = %d, want 2", p.PartCount)
	}
	if p.BytesOnDisk != 150 {
		t.Errorf("orphan BytesOnDisk = %d, want 150", p.BytesOnDisk)
	}

	// Sanity: repoE is reported incomplete, not as an orphan part.
	inc := r.IncompleteRegistered()
	if len(inc) != 1 || inc[0].ModelID != "org/repoE" {
		t.Errorf("IncompleteRegistered = %+v, want only org/repoE", inc)
	}
}

// Repos that keep each quant in a subfolder (unsloth's large models): the
// partial is named by its path in the repository. With the bare file name
// the Downloads panel could not match it to the download still running,
// and showed it a second time as an incomplete download to resume, whose
// Resume would have asked the host for a file that is not there.
func TestOrphanPartsInASubfolder(t *testing.T) {
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models")
	r := NewRegistry(dir, modelsDir)

	q := filepath.Join(modelsDir, "unsloth--Big-GGUF", "UD-Q4_K_XL")
	touch(t, filepath.Join(q, "Big-UD-Q4_K_XL-00001-of-00004.gguf"), 100)
	touch(t, filepath.Join(q, "Big-UD-Q4_K_XL-00002-of-00004.gguf.part"), 40)

	got := r.OrphanParts()
	if len(got) != 1 {
		t.Fatalf("OrphanParts = %+v", got)
	}
	if want := "UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00004.gguf"; got[0].Filename != want {
		t.Errorf("Filename = %q, want %q", got[0].Filename, want)
	}
	if got[0].ModelID != "unsloth/Big-GGUF" || got[0].BytesOnDisk != 140 {
		t.Errorf("orphan = %+v", got[0])
	}
}
