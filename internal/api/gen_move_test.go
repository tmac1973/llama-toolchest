package api

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

// Catalog files from the old location (next to the models dir) move into
// the models dir's gen-models folder; a file two sets share moves once
// and both sets follow it; a custom set's own files stay where they are.
func TestMoveGenModels(t *testing.T) {
	s := genServer(t)
	oldRoot := filepath.Join(s.cfg.DataDir, "gen-models")
	newRoot := generation.ModelsDir(s.cfg)

	write := func(path, body string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	vae := generation.Component{Slot: "vae", Repo: "Comfy-Org/z_image_turbo", File: "split_files/vae/ae.safetensors", Bytes: 3}
	vae.Path = huggingface.FilePath(oldRoot, vae.Repo, vae.File)
	write(vae.Path, "vae")
	var sets []generation.ModelSet
	for _, id := range []string{"z-image-turbo", "chroma1-hd"} {
		dit := generation.Component{Slot: "diffusion_model", Repo: "org/" + id, File: id + ".gguf", Bytes: 3}
		dit.Path = huggingface.FilePath(oldRoot, dit.Repo, dit.File)
		write(dit.Path, "dit")
		sets = append(sets, generation.ModelSet{ID: id, Name: id, Kind: generation.KindSD, CatalogID: id, Complete: true,
			Components: []generation.Component{dit, vae}})
	}
	custom := filepath.Join(s.cfg.DataDir, "mine", "model.gguf")
	write(custom, "own")
	sets = append(sets, generation.ModelSet{ID: "mine", Name: "mine", Kind: generation.KindSD, Complete: true,
		Components: []generation.Component{{Slot: "model", Path: custom, Bytes: 3}}})
	for _, set := range sets {
		if err := s.gen.AddModelSet(set); err != nil {
			t.Fatal(err)
		}
	}

	s.startGenMove()
	s.genMove.wait()
	if err := s.genMove.busy(); err != nil {
		t.Fatalf("still busy after the move: %v", err)
	}

	for _, id := range []string{"z-image-turbo", "chroma1-hd"} {
		set, _ := s.gen.ModelSet(id)
		for _, c := range set.Components {
			want := huggingface.FilePath(newRoot, c.Repo, c.File)
			if c.Path != want {
				t.Errorf("%s/%s points at %s, want %s", id, c.Slot, c.Path, want)
			}
			if _, err := os.Stat(c.Path); err != nil {
				t.Errorf("%s/%s: %v", id, c.Slot, err)
			}
		}
	}
	if set, _ := s.gen.ModelSet("mine"); set.Components[0].Path != custom {
		t.Errorf("custom set moved to %s", set.Components[0].Path)
	}
	if _, err := os.Stat(custom); err != nil {
		t.Errorf("custom file: %v", err)
	}
	if _, err := os.Stat(oldRoot); !os.IsNotExist(err) {
		t.Errorf("old gen-models folder left behind: %v", err)
	}
}

// Nothing to move: nothing is marked busy, so starting instances is not
// held up.
func TestMoveGenModelsNothingToDo(t *testing.T) {
	s := genServer(t)
	s.startGenMove()
	if err := s.genMove.busy(); err != nil {
		t.Errorf("busy with nothing to move: %v", err)
	}
	s.genMove.wait()
}

// A copy cut short leaves no file at the target that looks finished.
func TestCopyFileLeavesNoPartOnError(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "out.gguf")
	if err := copyFile(filepath.Join(dir, "missing.gguf"), dst); err == nil {
		t.Fatal("copying a missing file succeeded")
	}
	for _, p := range []string{dst, dst + ".part"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s exists after a failed copy", p)
		}
	}
}
