package api

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/modelsource"
)

const kleinRepo = "black-forest-labs/FLUX.2-klein-4B"

func browseServer(t *testing.T) (*Server, *hfFake) {
	t.Helper()
	s, fake, _ := linkServer(t, nil)
	fake.files[kleinRepo] = []modelsource.RepoFile{
		{Path: "flux-2-klein-4b.safetensors", Size: 7_750_000_000},
		{Path: "README.md", Size: 100},
		{Path: "model_index.json", Size: 500},
		{Path: "tokenizer/tokenizer.json", Size: 9_000_000},
	}
	return s, fake
}

func browse(t *testing.T, s *Server, q url.Values) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleGenSetBrowse(rec, htmxReq("GET", "/?"+q.Encode(), nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("browse %v: HTTP %d %s", q, rec.Code, rec.Body)
	}
	return rec.Body.String()
}

func TestGenSetBrowse(t *testing.T) {
	s, fake := browseServer(t)
	q := url.Values{"slot": {"diffusion_model"}, "kind": {"sd"}, "recipe": {"flux2-klein-4b"}}
	page := browse(t, s, q)
	for _, want := range []string{
		"Files in " + kleinRepo,                 // the recipe's repo is the default
		"flux-2-klein-4b.safetensors", "7.8 GB", // a model file, sized
		"https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/flux-2-klein-4b.safetensors",
		"Diffusers layout", // model_index.json
		"leejet/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf", // the catalog set's file
		"Show all files (3 more)",                               // README, model_index, tokenizer
	} {
		if !strings.Contains(page, want) {
			t.Errorf("browse page lacks %q", want)
		}
	}
	if strings.Contains(page, "README.md") {
		t.Error("the filter shows README.md")
	}

	q.Set("all", "1")
	if page := browse(t, s, q); !strings.Contains(page, "README.md") || !strings.Contains(page, "Show model files only") {
		t.Error("all=1 does not show every file")
	}

	tok := url.Values{"slot": {"tokenizer"}, "kind": {"sd"}, "recipe": {"flux2-klein-4b"}, "repo": {kleinRepo}}
	if page := browse(t, s, tok); !strings.Contains(page, "tokenizer/tokenizer.json") || strings.Contains(page, "flux-2-klein-4b.safetensors</code>") {
		t.Error("the tokenizer slot does not list JSON files only")
	}

	// A typed link to a file selects its repository.
	other := url.Values{"slot": {"upscaler"}, "kind": {"upscale"}, "repo": {"https://huggingface.co/org/up/blob/main/x4.pth"}}
	if page := browse(t, s, other); !strings.Contains(page, "Files in org/up") || !strings.Contains(page, "x4.pth") {
		t.Error("a pasted link did not list its repository")
	}

	// A failed listing is shown in the dialog, not as an HTTP error.
	gated := url.Values{"slot": {"vae"}, "kind": {"sd"}, "recipe": {"flux2-klein-4b"}, "repo": {"org/gated"}}
	s.hfRepoFake = &failingLister{hfFake: fake, err: errors.New("org/gated is a gated model")}
	if page := browse(t, s, gated); !strings.Contains(page, "org/gated is a gated model") {
		t.Errorf("the gated message is not in the dialog:\n%s", page)
	}
}

// failingLister fails every listing.
type failingLister struct {
	*hfFake
	err error
}

func (f *failingLister) RepoFilesAt(ctx context.Context, repo, rev string) ([]modelsource.RepoFile, error) {
	return nil, f.err
}

func TestGenSetFormSize(t *testing.T) {
	s, _ := browseServer(t)
	size := func(form url.Values) string {
		rec := httptest.NewRecorder()
		s.handleGenSetFormSize(rec, htmxReq("GET", "/?"+form.Encode(), nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("HTTP %d", rec.Code)
		}
		return rec.Body.String()
	}
	if got := size(url.Values{"slot_upscaler": {"/data/models/x4.pth"}}); got != "" {
		t.Errorf("local files only: %q, want nothing", got)
	}
	if got := size(url.Values{"slot_upscaler": {upLink}}); !strings.Contains(got, "Download: ") || strings.Contains(got, "del-color") {
		t.Errorf("one link: %q", got)
	}
	if got := size(url.Values{"slot_upscaler": {"https://huggingface.co/org/big/blob/main/huge.safetensors"}}); s.downloader.AvailableForDownload() >= 0 &&
		(!strings.Contains(got, "GB is free in the models folder") || !strings.Contains(got, "del-color")) {
		t.Errorf("too big: %q", got)
	}
	if got := size(url.Values{"slot_upscaler": {"https://huggingface.co/org/up/blob/main/nope.pth"}}); !strings.Contains(got, "nope.pth is not in org/up") {
		t.Errorf("missing file: %q", got)
	}
}
