package autoconfig

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

func readTestdata(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestTrimCardKeepsRunningSections(t *testing.T) {
	out := TrimCard(readTestdata(t, "unsloth_card.md"))
	for _, want := range []string{"temperature=0.6", "llama-server", "draft-mtp"} {
		if !strings.Contains(out, want) {
			t.Errorf("trimmed card lost %q:\n%s", want, out)
		}
	}
	for _, gone := range []string{"license:", "<img", "shields.io", "Our company", "<div"} {
		if strings.Contains(out, gone) {
			t.Errorf("trimmed card still holds %q:\n%s", gone, out)
		}
	}
	if got := TrimCard(readTestdata(t, "no_llamacpp_card.md")); got != "" {
		t.Errorf("a card with nothing about running the model kept:\n%s", got)
	}
}

func TestCapTextCutsAtParagraph(t *testing.T) {
	text := strings.Repeat("a", 60) + "\n\n" + strings.Repeat("b", 60)
	out := capText(text, 100)
	if strings.Contains(out, "b") || !strings.Contains(out, "left out for length") {
		t.Errorf("capText = %q", out)
	}
	if capText("short", 100) != "short" {
		t.Error("text under the cap was changed")
	}
}

type fakeFetcher struct {
	pages map[string]string
	base  string
	urls  []string
	auth  []bool
}

func (f *fakeFetcher) CachedGet(_ context.Context, url string, withAuth bool) ([]byte, error) {
	f.urls = append(f.urls, url)
	f.auth = append(f.auth, withAuth)
	if p, ok := f.pages[url]; ok {
		return []byte(p), nil
	}
	return nil, errors.New("404")
}

func (f *fakeFetcher) ResolveBaseRepo(context.Context, string) string { return f.base }

func TestFetchCardReadsBothRepositories(t *testing.T) {
	f := &fakeFetcher{base: "Qwen/Qwen3.5-9B", pages: map[string]string{
		"https://hf.test/unsloth/Qwen3.5-9B-GGUF/raw/main/README.md": readTestdata(t, "unsloth_card.md"),
		"https://hf.test/Qwen/Qwen3.5-9B/raw/main/README.md":         readTestdata(t, "qwen_base_card.md"),
	}}
	m := &models.Model{ModelID: "unsloth/Qwen3.5-9B-GGUF"}
	card := FetchCard(context.Background(), f, "https://hf.test", m, 0)
	if len(card.Sources) != 2 || card.Sources[0] != "unsloth/Qwen3.5-9B-GGUF" {
		t.Fatalf("sources = %v", card.Sources)
	}
	gguf := strings.Index(card.Text, "temperature=0.6")
	base := strings.Index(card.Text, "32,768 tokens")
	if gguf < 0 || base < 0 || gguf > base {
		t.Errorf("the GGUF repository's card should come first:\n%s", card.Text)
	}
	for _, a := range f.auth {
		if !a {
			t.Error("model card fetched without the token (gated repositories need it)")
		}
	}
}

func TestFetchCardMissingIsEmpty(t *testing.T) {
	card := FetchCard(context.Background(), &fakeFetcher{}, "https://hf.test", &models.Model{ModelID: "a/b"}, 0)
	if card.Text != "" || len(card.Sources) != 0 {
		t.Errorf("missing card = %+v", card)
	}
	if card := FetchCard(context.Background(), &fakeFetcher{}, "https://hf.test", &models.Model{ModelID: "a/b", Source: "modelscope"}, 0); card.Text != "" {
		t.Error("a ModelScope model's card was read from Hugging Face")
	}
}

func TestCardCharsForContext(t *testing.T) {
	if got := CardCharsForContext(16384); got < 30000 || got > 48000 {
		t.Errorf("16K context gives %d characters", got)
	}
	if got := CardCharsForContext(2048); got != 4000 {
		t.Errorf("tiny context gives %d, want the 4000 floor", got)
	}
}
