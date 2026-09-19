package backup

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// Profiles travel through export and restore with the model configs:
// paths relative to the models dir on the way out, resolved on the way
// in, and a profile whose model is not installed is kept all the same.
func TestProfilesRoundTrip(t *testing.T) {
	cfg, b, reg := testState(t)
	id := "org--repo-GGUF--model-Q4_K_M"
	mmproj := filepath.Join(cfg.ModelsPath(), "org--repo-GGUF", "mmproj-BF16.gguf")
	c, _ := reg.GetConfig(id)
	c.ContextSize = 32768
	reg.SetConfig(id, c)
	if _, err := reg.SaveProfile(id, "Long context", models.ProfileSourceUser, "b1"); err != nil {
		t.Fatal(err)
	}
	if _, err := reg.ImportProfile(models.ConfigProfile{
		RepoID: "org/other-GGUF", Filename: "other-Q8_0.gguf", Name: "Not installed",
		Config: models.ModelConfig{ContextSize: 8192},
	}); err != nil {
		t.Fatal(err)
	}

	f := Assemble(cfg, b, reg, nil, false)
	if len(f.Profiles) != 2 {
		t.Fatalf("exported %d profiles, want 2", len(f.Profiles))
	}
	var exported models.ConfigProfile
	for _, p := range f.Profiles {
		if p.Name == "Long context" {
			exported = p
		}
	}
	if exported.Config.MmprojPath != filepath.Join("org--repo-GGUF", "mmproj-BF16.gguf") {
		t.Errorf("profile mmproj not relativized: %q", exported.Config.MmprojPath)
	}

	data, err := f.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}

	// Restore into a fresh machine that has the first model's mmproj file.
	target := models.NewRegistry(t.TempDir(), cfg.ModelsPath())
	if err := os.MkdirAll(filepath.Dir(mmproj), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mmproj, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	rec := newRecorder(1, cfg.ModelsPath())
	rec.deps.ImportProfile = func(p models.ConfigProfile) error {
		_, err := target.ImportProfile(p)
		return err
	}
	rep := Apply(parsed, Selections{ModelConfigs: true}, rec.deps)
	if rep.AppliedProfiles != 2 {
		t.Fatalf("applied %d profiles, want 2; report %+v", rep.AppliedProfiles, rep)
	}

	// The profile shows up once its model registers on the target.
	if err := target.Add(&models.Model{ID: "scanned-id", ModelID: "org/repo-GGUF", Filename: "model-Q4_K_M.gguf", Quant: "Q4_K_M"}); err != nil {
		t.Fatal(err)
	}
	got, err := target.GetProfile("scanned-id", "long context")
	if err != nil {
		t.Fatal(err)
	}
	if got.Config.ContextSize != 32768 || got.BuildID != "b1" || got.Config.MmprojPath != mmproj {
		t.Errorf("restored profile = ctx %d build %q mmproj %q", got.Config.ContextSize, got.BuildID, got.Config.MmprojPath)
	}
	if n := len(target.AllProfiles()); n != 2 {
		t.Errorf("target holds %d profiles, want 2 (including the one for the missing model)", n)
	}
}

// Unticking model configs also leaves the profiles alone, and says so.
func TestProfilesNotSelected(t *testing.T) {
	f := &File{Version: Version, Profiles: []models.ConfigProfile{{RepoID: "a/b", Filename: "c.gguf", Name: "p"}}}
	rec := newRecorder(1, "")
	imported := 0
	rec.deps.ImportProfile = func(models.ConfigProfile) error { imported++; return nil }
	rep := Apply(f, Selections{Settings: true}, rec.deps)
	if imported != 0 {
		t.Errorf("imported %d profiles with model configs unticked", imported)
	}
	if len(rep.NotSelected) != 1 || rep.NotSelected[0] != "model configs" {
		t.Errorf("NotSelected = %v, want [model configs]", rep.NotSelected)
	}
}
