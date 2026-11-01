package generation

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCheckFileFormat(t *testing.T) {
	dir := t.TempDir()
	header := []byte(`{"a":1}`)
	safetensors := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	safetensors = append(append(safetensors, header...), make([]byte, 16)...)
	files := map[string][]byte{
		"model.gguf":        []byte("GGUF\x03\x00\x00\x00rest"),
		"vae.safetensors":   safetensors,
		"esrgan.pth":        []byte("PK\x03\x04rest of zip"),
		"old.ckpt":          {0x80, 0x02, 'x'},
		"tokenizer.json":    []byte("\n  {\"model\": {}}"),
		"notes.txt":         []byte("just some text"),
		"short.safetensors": append(binary.LittleEndian.AppendUint64(nil, 1<<40), '{'), // header longer than the file
	}
	for name, data := range files {
		writeFile(t, filepath.Join(dir, name), data)
	}
	for _, tc := range []struct {
		slot, file string
		ok         bool
	}{
		{"diffusion_model", "model.gguf", true},
		{"vae", "vae.safetensors", true},
		{"upscaler", "esrgan.pth", true},
		{"model", "old.ckpt", true},
		{"tokenizer", "tokenizer.json", true},
		{"diffusion_model", "notes.txt", false},
		{"diffusion_model", "tokenizer.json", false},
		{"tokenizer", "model.gguf", false},
		{"vae", "short.safetensors", false},
	} {
		err := CheckFileFormat(tc.slot, filepath.Join(dir, tc.file))
		if (err == nil) != tc.ok {
			t.Errorf("%s in %s: err = %v, want ok %v", tc.file, tc.slot, err, tc.ok)
		}
	}
	if err := CheckFileFormat("vae", filepath.Join(dir, "notes.txt")); err == nil || !strings.Contains(err.Error(), "is not a model file (GGUF, safetensors or PyTorch)") {
		t.Errorf("message = %v", err)
	}
}

// A missing file says where the models folder is, and in a container that
// host folders are not visible.
func TestCheckFilesMissingMessage(t *testing.T) {
	comps := []Component{{Slot: "vae", Path: "/home/me/vae.safetensors"}}
	err := CheckFiles(comps, "/data/models/gen-models", false)
	if err == nil || !strings.Contains(err.Error(), "does not exist. The models folder is /data/models/gen-models.") || strings.Contains(err.Error(), "container") {
		t.Errorf("host: %v", err)
	}
	err = CheckFiles(comps, "/data/models/gen-models", true)
	if err == nil || !strings.Contains(err.Error(), "runs in a container and cannot see host folders") {
		t.Errorf("container: %v", err)
	}
}

func TestListModelFiles(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{
		"a/model.gguf", "a/vae.safetensors", "b/x4.pth", "b/old.ckpt",
		"a/big.gguf.part", "a/config.json", "a/mllm/tokenizer.json", "a/readme.md",
		".cache/hidden.gguf",
	} {
		writeFile(t, filepath.Join(dir, name), []byte("x"))
	}
	outside := t.TempDir()
	writeFile(t, filepath.Join(outside, "linked.gguf"), []byte("x"))
	if err := os.Symlink(outside, filepath.Join(dir, "linked-folder")); err != nil {
		t.Fatal(err)
	}

	models, tokenizers, truncated := ListModelFiles(dir, 100)
	var rel []string
	for _, p := range models {
		r, _ := filepath.Rel(dir, p)
		rel = append(rel, r)
	}
	if want := []string{"a/model.gguf", "a/vae.safetensors", "b/old.ckpt", "b/x4.pth"}; !slices.Equal(rel, want) {
		t.Errorf("models = %v, want %v", rel, want)
	}
	if len(tokenizers) != 1 || !strings.HasSuffix(tokenizers[0], "a/mllm/tokenizer.json") || truncated {
		t.Errorf("tokenizers = %v, truncated %v", tokenizers, truncated)
	}

	models, tokenizers, truncated = ListModelFiles(dir, 2)
	if len(models)+len(tokenizers) != 2 || !truncated {
		t.Errorf("max 2: %d files, truncated %v", len(models)+len(tokenizers), truncated)
	}
	if _, _, truncated := ListModelFiles(dir, 5); truncated {
		t.Error("exactly max files is not truncated")
	}
}
