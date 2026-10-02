package builder

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestParseArchNames(t *testing.T) {
	src, err := os.ReadFile("testdata/llama-arch.cpp")
	if err != nil {
		t.Fatal(err)
	}
	names := ParseArchNames(src)
	// 154 entries, less the "clip" placeholder and "(unknown)".
	if len(names) != 152 {
		t.Errorf("%d names, want 152", len(names))
	}
	for _, want := range []string{"llama", "qwen3moe", "gemma3", "gemma4", "gpt-oss", "deepseek2"} {
		if !slices.Contains(names, want) {
			t.Errorf("%q missing", want)
		}
	}
	for _, unwanted := range []string{"clip", "(unknown)"} {
		if slices.Contains(names, unwanted) {
			t.Errorf("%q listed", unwanted)
		}
	}
}

// A file without the table is "not known", never "nothing supported".
func TestParseArchNamesWithoutTheTable(t *testing.T) {
	if names := ParseArchNames([]byte("int main() { return 0; }")); names != nil {
		t.Errorf("names = %v, want nil", names)
	}
}

// Builds from before the list was recorded get it from the checkout's
// history, at the commit they were built from.
func TestBackfillArchs(t *testing.T) {
	dataDir, srcDir := gitFixture(t)
	git := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-C", srcDir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1",
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	if err := os.MkdirAll(filepath.Join(srcDir, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	src := `static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA, "llama" },
    { LLM_ARCH_QWEN3, "qwen3" },
};`
	if err := os.WriteFile(filepath.Join(srcDir, archSourceFile), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", archSourceFile)
	git("commit", "-q", "-m", "arch", "--no-gpg-sign")
	sha := git("rev-parse", "HEAD")

	b := &Builder{dataDir: dataDir, builds: []BuildResult{
		{ID: "old", GitSHA: sha, Status: BuildStatusSuccess},
		{ID: "gone", GitSHA: "0123456789abcdef0123456789abcdef01234567", Status: BuildStatusSuccess},
		{ID: "failed", GitSHA: sha, Status: BuildStatusFailed},
		{ID: "recorded", GitSHA: sha, Status: BuildStatusSuccess, Archs: []string{"mamba"}},
	}}
	b.BackfillArchs()

	got := map[string][]string{}
	for _, br := range b.List() {
		got[br.ID] = br.Archs
	}
	if !slices.Equal(got["old"], []string{"llama", "qwen3"}) {
		t.Errorf("old = %v, want [llama qwen3]", got["old"])
	}
	if got["gone"] != nil || got["failed"] != nil {
		t.Errorf("gone = %v, failed = %v; want both empty", got["gone"], got["failed"])
	}
	if !slices.Equal(got["recorded"], []string{"mamba"}) {
		t.Errorf("a recorded list was replaced: %v", got["recorded"])
	}

	// And it was saved.
	reloaded := NewBuilder(dataDir)
	if br, ok := reloaded.Find("old"); !ok || len(br.Archs) != 2 {
		t.Errorf("not saved: %+v", br)
	}
}
