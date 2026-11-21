package generation

import (
	"encoding/json"
	"testing"
)

func TestParseHFLink(t *testing.T) {
	cases := []struct {
		in              string
		repo, rev, file string
		ok              bool
	}{
		{"https://huggingface.co/stabilityai/sd-turbo/blob/main/sd_turbo.safetensors", "stabilityai/sd-turbo", "main", "sd_turbo.safetensors", true},
		{"https://huggingface.co/org/repo/resolve/0123abc/model.gguf?download=true", "org/repo", "0123abc", "model.gguf", true},
		{"  https://www.huggingface.co/org/repo/blob/main/x.safetensors#top ", "org/repo", "main", "x.safetensors", true},
		{"http://huggingface.co/gpt2/blob/main/model.safetensors", "gpt2", "main", "model.safetensors", true},
		{"https://huggingface.co/Comfy-Org/z/blob/main/split_files/vae/ae.safetensors", "Comfy-Org/z", "main", "split_files/vae/ae.safetensors", true},
		{"https://huggingface.co/org/repo/blob/main/my%20model.safetensors", "org/repo", "main", "my model.safetensors", true},
		{"https://huggingface.co/org/repo", "", "", "", false},
		{"https://huggingface.co/org/repo/tree/main", "", "", "", false},
		{"https://huggingface.co/org/repo/blob/main", "", "", "", false},
		{"https://huggingface.co/a/b/c/blob/main/x", "", "", "", false},
		{"https://example.com/org/repo/blob/main/x.safetensors", "", "", "", false},
		{"/data/models/gen-test/sd_turbo.safetensors", "", "", "", false},
		{"", "", "", "", false},
	}
	for _, c := range cases {
		repo, rev, file, ok := ParseHFLink(c.in)
		if repo != c.repo || rev != c.rev || file != c.file || ok != c.ok {
			t.Errorf("ParseHFLink(%q) = %q, %q, %q, %v; want %q, %q, %q, %v",
				c.in, repo, rev, file, ok, c.repo, c.rev, c.file, c.ok)
		}
	}
}

func TestHFLinkRoundTrip(t *testing.T) {
	for _, c := range []struct{ repo, rev, file, want string }{
		{"org/repo", "", "a.safetensors", "https://huggingface.co/org/repo/blob/main/a.safetensors"},
		{"org/repo", "0123", "split_files/vae/my vae.safetensors", "https://huggingface.co/org/repo/blob/0123/split_files/vae/my%20vae.safetensors"},
		{"gpt2", "main", "m.gguf", "https://huggingface.co/gpt2/blob/main/m.gguf"},
	} {
		link := HFLink(c.repo, c.rev, c.file)
		if link != c.want {
			t.Errorf("HFLink(%q, %q, %q) = %q, want %q", c.repo, c.rev, c.file, link, c.want)
		}
		wantRev := c.rev
		if wantRev == "" {
			wantRev = "main"
		}
		repo, rev, file, ok := ParseHFLink(link)
		if !ok || repo != c.repo || rev != wantRev || file != c.file {
			t.Errorf("ParseHFLink(%q) = %q, %q, %q, %v", link, repo, rev, file, ok)
		}
	}
}

func TestRecipeModelRepo(t *testing.T) {
	for url, want := range map[string]string{
		"https://huggingface.co/black-forest-labs/FLUX.2-klein-4B": "black-forest-labs/FLUX.2-klein-4B",
		"https://huggingface.co/org/name/":                         "org/name",
		"https://huggingface.co/org/name/tree/main":                "org/name",
		"https://huggingface.co/org/name/blob/main/x.safetensors":  "",
		"https://github.com/leejet/stable-diffusion.cpp":           "",
		"": "",
	} {
		r := &Recipe{ModelURL: url}
		if got := r.ModelRepo(); got != want {
			t.Errorf("ModelRepo(%q) = %q, want %q", url, got, want)
		}
	}
}

func TestSnapshotModelRepos(t *testing.T) {
	book := SnapshotRecipes()
	r, ok := book.Get("flux2-klein-4b")
	if !ok {
		t.Fatal("no flux2-klein-4b recipe")
	}
	if got := r.ModelRepo(); got != "black-forest-labs/FLUX.2-klein-4B" {
		t.Errorf("flux2-klein-4b ModelRepo = %q", got)
	}
	n := 0
	for _, kind := range []BackendKind{KindSD, KindVideo} {
		for _, id := range book.IDs(kind) {
			if r, _ := book.Get(id); r.ModelRepo() != "" {
				n++
			}
		}
		for _, r := range book.Unavailable(kind) {
			if r.ModelRepo() != "" {
				n++
			}
		}
	}
	// 104 of the 108 snapshot recipes link a Hugging Face repo. Far fewer
	// means model_url is no longer read.
	if n < 100 {
		t.Errorf("only %d snapshot recipes have a Hugging Face model repo", n)
	}
}

// Sets saved before revisions were pinned load as they are, and saving
// them again adds no revision or error keys.
func TestComponentJSONKeepsOldShape(t *testing.T) {
	old := `{"slot":"vae","repo":"org/repo","file":"ae.safetensors","path":"/data/models/gen/org--repo/ae.safetensors","bytes":42}`
	var c Component
	if err := json.Unmarshal([]byte(old), &c); err != nil {
		t.Fatal(err)
	}
	if c.Revision != "" || c.Error != "" || c.Bytes != 42 {
		t.Errorf("loaded %+v", c)
	}
	out, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != old {
		t.Errorf("saved %s, want %s", out, old)
	}
}
