package models

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// newTestRegistry returns an empty registry whose data and models folders
// sit in a temporary directory.
func newTestRegistry(t *testing.T) (r *Registry, dataDir, modelsDir string) {
	t.Helper()
	dataDir = t.TempDir()
	modelsDir = filepath.Join(dataDir, "models")
	return NewRegistry(dataDir, modelsDir), dataDir, modelsDir
}

func modelIDs(ms []*Model) []string {
	var out []string
	for _, m := range ms {
		out = append(out, m.ID)
	}
	return out
}

// Every /api/models/{id} handler passes the path value through ResolveID,
// so a model must be reachable by its registry ID, its public /v1 name
// and any alias the user gave it. A name that matches nothing comes back
// unchanged, so the handler's "not found" error names what the user typed.
func TestResolveID(t *testing.T) {
	r, _, _ := newTestRegistry(t)
	m := &Model{
		ID:       "unsloth--Qwen3-8B-GGUF--Qwen3-8B-Q4_K_M",
		ModelID:  "unsloth/Qwen3-8B-GGUF",
		Filename: "Qwen3-8B-Q4_K_M.gguf",
		Quant:    "Q4_K_M",
	}
	if err := r.Add(m); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig()
	cfg.Aliases = []string{"my-qwen"}
	if err := r.SetConfig(m.ID, &cfg); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name, in, want string
	}{
		{"registry ID", m.ID, m.ID},
		{"public name", "unsloth-Qwen3-8B.Q4_K_M", m.ID},
		{"alias", "my-qwen", m.ID},
		{"unknown name", "no-such-model", "no-such-model"},
		{"empty name", "", ""},
		{"HuggingFace repo alone is not a name", "unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := r.ResolveID(tt.in); got != tt.want {
				t.Errorf("ResolveID(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// FindOrphans drives the "file missing" notice and its clean-up action:
// it lists exactly the records whose file is gone from disk, and leaves
// alone the ones whose file is still there.
func TestFindOrphans(t *testing.T) {
	r, _, modelsDir := newTestRegistry(t)

	touch(t, filepath.Join(modelsDir, "org--present", "present.gguf"), 10)
	register(t, r, modelsDir, "org/present", "present.gguf")
	register(t, r, modelsDir, "org/gone", "gone.gguf")

	got := modelIDs(r.FindOrphans())
	if want := []string{"org/gone--gone.gguf"}; !slices.Equal(got, want) {
		t.Errorf("FindOrphans = %v, want %v", got, want)
	}

	// The file coming back clears it.
	touch(t, filepath.Join(modelsDir, "org--gone", "gone.gguf"), 10)
	if got := r.FindOrphans(); len(got) != 0 {
		t.Errorf("FindOrphans after restore = %v, want none", modelIDs(got))
	}
}

// The disk scan registers a split model once, by its first shard, and
// skips the rest. A file name that only looks a little like a shard name
// must still be registered.
func TestIsNonFirstShard(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"model-00001-of-00003.gguf", false},
		{"model-00002-of-00003.gguf", true},
		{"model-00003-of-00003.gguf", true},
		{"model.gguf", false},
		{"model-Q4_K_M.gguf", false},
		{"model-2-of-3.gguf", false},               // not five digits
		{"model-00002-of-00003.gguf.part", false},  // a partial download, not a shard
		{"model-00002-of-00003-extra.gguf", false}, // the shard part must end the name
		{"UD-Q4_K_XL-00010-of-00012.gguf", true},
	}
	for _, tt := range tests {
		if got := isNonFirstShard(tt.name); got != tt.want {
			t.Errorf("isNonFirstShard(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// findShards decides which files Delete removes and which sizes the scan
// adds up, so it must name every part of a split model whichever part it
// is given, and only the file itself for a model in one file.
func TestFindShards(t *testing.T) {
	dir := filepath.Join("models", "org--repo")
	tests := []struct {
		name     string
		filename string
		want     []string
	}{
		{"single file", "model.gguf", []string{"model.gguf"}},
		{"first of three", "big-00001-of-00003.gguf",
			[]string{"big-00001-of-00003.gguf", "big-00002-of-00003.gguf", "big-00003-of-00003.gguf"}},
		{"middle of three gives the same set", "big-00002-of-00003.gguf",
			[]string{"big-00001-of-00003.gguf", "big-00002-of-00003.gguf", "big-00003-of-00003.gguf"}},
		{"a set of one is a single file", "solo-00001-of-00001.gguf", []string{"solo-00001-of-00001.gguf"}},
		{"name with dashes before the shard part", "Qwen3-8B-Q4_K_M-00001-of-00002.gguf",
			[]string{"Qwen3-8B-Q4_K_M-00001-of-00002.gguf", "Qwen3-8B-Q4_K_M-00002-of-00002.gguf"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var want []string
			for _, f := range tt.want {
				want = append(want, filepath.Join(dir, f))
			}
			if got := findShards(dir, tt.filename); !slices.Equal(got, want) {
				t.Errorf("findShards = %v, want %v", got, want)
			}
		})
	}
}

// The vision projector is found next to the model, or one folder up for
// repos that keep each quant in a subfolder. Only a .gguf file counts,
// and a folder whose name contains "mmproj" is not a projector.
func TestFindMMProj(t *testing.T) {
	tests := []struct {
		name  string
		files []string // relative to the repo folder
		model string   // relative to the repo folder
		want  string   // relative to the repo folder; "" for none
	}{
		{"next to the model", []string{"model.gguf", "mmproj-F16.gguf"}, "model.gguf", "mmproj-F16.gguf"},
		{"case does not matter", []string{"model.gguf", "MMPROJ-BF16.GGUF"}, "model.gguf", "MMPROJ-BF16.GGUF"},
		{"one folder up", []string{"Q4_K_M/model.gguf", "mmproj-F16.gguf"}, "Q4_K_M/model.gguf", "mmproj-F16.gguf"},
		{"next to the model wins over one folder up",
			[]string{"Q4_K_M/model.gguf", "Q4_K_M/mmproj-near.gguf", "mmproj-far.gguf"}, "Q4_K_M/model.gguf", "Q4_K_M/mmproj-near.gguf"},
		{"not a .gguf file", []string{"model.gguf", "mmproj-F16.gguf.part"}, "model.gguf", ""},
		{"a folder named like a projector", []string{"model.gguf", "mmproj.gguf/readme.txt"}, "model.gguf", ""},
		{"none", []string{"model.gguf"}, "model.gguf", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := filepath.Join(t.TempDir(), "org--repo")
			for _, f := range tt.files {
				touch(t, filepath.Join(repo, f), 1)
			}
			want := ""
			if tt.want != "" {
				want = filepath.Join(repo, tt.want)
			}
			if got := FindMMProj(filepath.Join(repo, tt.model)); got != want {
				t.Errorf("FindMMProj = %q, want %q", got, want)
			}
		})
	}
}

// A separate MTP drafter head is found next to the model, in an "MTP"
// subfolder, or the same places one folder up. The file's architecture
// decides: a runnable self-speculation model with "MTP" in its name is
// not a head, and a file with neither "mtp" nor "assistant" in its name
// is only opened when it sits in an "MTP" folder.
func TestFindMTP(t *testing.T) {
	head := readFile(t, writeMTPGGUF(t, "gemma4-assistant", 0, []string{"blk.0.attn_q.weight"}))
	// A qwen3 file with built-in MTP layers and its own block 0: a model
	// that drafts for itself, not a head.
	runnable := readFile(t, writeMTPGGUF(t, "qwen3", 1, []string{"blk.0.attn_q.weight", "blk.36.attn_q.weight"}))

	tests := []struct {
		name  string
		files map[string][]byte // relative to the repo folder
		model string
		want  string
	}{
		{"head next to the model",
			map[string][]byte{"model.gguf": runnable, "gemma-4-mtp-Q8_0.gguf": head}, "model.gguf", "gemma-4-mtp-Q8_0.gguf"},
		{"head named assistant",
			map[string][]byte{"model.gguf": runnable, "gemma-4-assistant.gguf": head}, "model.gguf", "gemma-4-assistant.gguf"},
		{"head in an MTP subfolder under any name",
			map[string][]byte{"model.gguf": runnable, "MTP/drafter.gguf": head}, "model.gguf", "MTP/drafter.gguf"},
		{"head in the MTP folder one level up",
			map[string][]byte{"Q4_K_M/model.gguf": runnable, "MTP/drafter.gguf": head}, "Q4_K_M/model.gguf", "MTP/drafter.gguf"},
		{"runnable model with MTP in its name is not a head",
			map[string][]byte{"model.gguf": runnable, "Qwen3-8B-MTP-Q4_K_M.gguf": runnable}, "model.gguf", ""},
		{"head without a telling name, outside an MTP folder, is not opened",
			map[string][]byte{"model.gguf": runnable, "drafter.gguf": head}, "model.gguf", ""},
		{"a file that is not a GGUF", map[string][]byte{"model.gguf": runnable, "notes-mtp.gguf": []byte("text")}, "model.gguf", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo := filepath.Join(t.TempDir(), "org--repo")
			for rel, data := range tt.files {
				p := filepath.Join(repo, rel)
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(p, data, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			want := ""
			if tt.want != "" {
				want = filepath.Join(repo, tt.want)
			}
			if got := FindMTP(filepath.Join(repo, tt.model)); got != want {
				t.Errorf("FindMTP = %q, want %q", got, want)
			}
		})
	}
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// After a delete, removeEmptyDirs walks up from the model's folder and
// removes each folder that is now empty, stopping at the first one that
// still holds something. It must never remove a folder with files in it.
//
// The models folder here always holds another model. The function's
// comment says it stops at the models folder, but it is not told where
// that folder is, so an empty models folder is removed too (and so are
// empty folders above it). That case is left out until it is fixed.
func TestRemoveEmptyDirs(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	touch(t, filepath.Join(models, "org--other", "other.gguf"), 1)

	// Two empty levels under a non-empty models folder: both go, the
	// models folder and the other repo stay.
	quant := filepath.Join(models, "org--repo", "Q4_K_M")
	if err := os.MkdirAll(quant, 0o755); err != nil {
		t.Fatal(err)
	}
	removeEmptyDirs(quant)
	if exists(quant) || exists(filepath.Join(models, "org--repo")) {
		t.Error("empty folders were left behind")
	}
	if !exists(filepath.Join(models, "org--other", "other.gguf")) {
		t.Error("a folder with a model in it was removed")
	}

	// A folder that still holds a file is left as it is, and so is
	// everything above it.
	keep := filepath.Join(models, "org--keep", "Q8_0")
	touch(t, filepath.Join(keep, "mmproj.gguf"), 1)
	removeEmptyDirs(keep)
	if !exists(filepath.Join(keep, "mmproj.gguf")) {
		t.Error("a folder with a file in it was removed")
	}

	// An empty sibling folder is not touched: only the given folder and
	// its parents are considered.
	sibling := filepath.Join(models, "org--keep", "empty-sibling")
	if err := os.MkdirAll(sibling, 0o755); err != nil {
		t.Fatal(err)
	}
	removeEmptyDirs(keep)
	if !exists(sibling) {
		t.Error("an empty sibling folder was removed")
	}

	// A folder that does not exist is not an error.
	removeEmptyDirs(filepath.Join(models, "never-made"))
}

// Delete removes every part of a split model, any partial download of a
// part, and the repo folder once it is empty. Other models and the models
// folder stay.
func TestDeleteRemovesShardsAndEmptyFolder(t *testing.T) {
	r, _, modelsDir := newTestRegistry(t)
	repo := filepath.Join(modelsDir, "org--split")
	touch(t, filepath.Join(repo, "big-00001-of-00002.gguf"), 10)
	touch(t, filepath.Join(repo, "big-00002-of-00002.gguf"), 10)
	touch(t, filepath.Join(repo, "big-00002-of-00002.gguf.part"), 5)
	register(t, r, modelsDir, "org/split", "big-00001-of-00002.gguf")

	touch(t, filepath.Join(modelsDir, "org--other", "other.gguf"), 10)
	register(t, r, modelsDir, "org/other", "other.gguf")

	if err := r.Delete("org/split--big-00001-of-00002.gguf"); err != nil {
		t.Fatal(err)
	}
	if exists(repo) {
		entries, _ := os.ReadDir(repo)
		t.Errorf("repo folder left behind, holding %d entries", len(entries))
	}
	if !exists(filepath.Join(modelsDir, "org--other", "other.gguf")) {
		t.Error("another model's file was removed")
	}
	if got := modelIDs(r.List()); !slices.Equal(got, []string{"org/other--other.gguf"}) {
		t.Errorf("registry after delete = %v", got)
	}
	if err := r.Delete("org/split--big-00001-of-00002.gguf"); err == nil {
		t.Error("deleting a model that is gone: no error")
	}
}

// A helper model is one the app downloaded for its own use. Marking one
// takes it out of every list a user picks models from (chat, embeddings,
// benchmarks — all fed by ListServing), puts it in ListHelpers, and
// survives a restart.
func TestSetHelperRole(t *testing.T) {
	r, dataDir, modelsDir := newTestRegistry(t)
	register(t, r, modelsDir, "org/b-chat", "chat.gguf")
	register(t, r, modelsDir, "org/a-helper", "helper.gguf")
	register(t, r, modelsDir, "org/c-chat", "chat.gguf")
	helper := "org/a-helper--helper.gguf"

	if err := r.SetHelperRole("no-such-model", true); err == nil {
		t.Error("unknown model: no error")
	}

	// Before: every model is served, in ModelID order.
	all := []string{helper, "org/b-chat--chat.gguf", "org/c-chat--chat.gguf"}
	if got := modelIDs(r.ListServing()); !slices.Equal(got, all) {
		t.Errorf("ListServing = %v, want %v", got, all)
	}

	if err := r.SetHelperRole(helper, true); err != nil {
		t.Fatal(err)
	}
	// Setting it again is not an error.
	if err := r.SetHelperRole(helper, true); err != nil {
		t.Fatal(err)
	}
	served := []string{"org/b-chat--chat.gguf", "org/c-chat--chat.gguf"}
	check := func(label string, r *Registry) {
		t.Helper()
		if got := modelIDs(r.ListServing()); !slices.Equal(got, served) {
			t.Errorf("%s: ListServing = %v, want %v", label, got, served)
		}
		if got := modelIDs(r.ListHelpers()); !slices.Equal(got, []string{helper}) {
			t.Errorf("%s: ListHelpers = %v, want [%s]", label, got, helper)
		}
	}
	check("after marking", r)
	check("after reload", NewRegistry(dataDir, modelsDir))

	// Unmarking puts it back.
	if err := r.SetHelperRole(helper, false); err != nil {
		t.Fatal(err)
	}
	if got := modelIDs(r.ListServing()); !slices.Equal(got, all) {
		t.Errorf("after unmarking: ListServing = %v, want %v", got, all)
	}
	if got := r.ListHelpers(); len(got) != 0 {
		t.Errorf("after unmarking: ListHelpers = %v", modelIDs(got))
	}
}
