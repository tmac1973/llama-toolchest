package builder

import (
	"slices"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/forks"
)

// Each curated fork is a project built like llama.cpp from its own repo.
func TestForkProjects(t *testing.T) {
	llama, _ := FindProject(ProjectLlamaCpp)
	for _, f := range forks.Curated() {
		p, ok := FindProject(f.ID)
		if !ok {
			t.Fatalf("no project for fork %s", f.ID)
		}
		if !p.IsFork() || p.Repo != f.Repo || p.IDPrefix != f.ID+"-" || p.CloneDir != f.ID {
			t.Errorf("%s project = %+v", f.ID, p)
		}
		if p.Required.Name != "llama-server" || !slices.Equal(p.BaseCMakeArgs, llama.BaseCMakeArgs) {
			t.Errorf("%s does not build llama-server like llama.cpp", f.ID)
		}
		for _, b := range []string{"rocm", "cuda"} {
			if p.SupportsBackend(b) != slices.Contains(f.Backends, b) {
				t.Errorf("%s SupportsBackend(%s) disagrees with the fork entry", f.ID, b)
			}
		}
		refs, err := p.listRefs(t.Context(), "")
		if err != nil || !slices.Equal(refs, []string{f.PinnedSHA}) {
			t.Errorf("%s refs = %v, %v; want the pinned commit only", f.ID, refs, err)
		}
		// A fork takes llama.cpp's options, LLAMA_* included.
		if !slices.EqualFunc(ProjectOptions(f.ID, "rocm"), ProjectOptions(ProjectLlamaCpp, "rocm"),
			func(a, b BuildOption) bool { return a.Flag == b.Flag }) {
			t.Errorf("%s options differ from llama.cpp's", f.ID)
		}
	}
	avalon, ok := FindProject("avalon")
	if !ok || avalon.SupportsBackend("vulkan") {
		t.Errorf("avalon project = %v, vulkan %v", ok, avalon.SupportsBackend("vulkan"))
	}
	// A generation kind never finds a fork.
	if p, ok := ProjectForKind(ForkKind); ok {
		t.Errorf("ProjectForKind(fork) = %s", p.ID)
	}
}

func TestBuildIDRefShortensFullSHA(t *testing.T) {
	sha := "ad2c6e750c7e1fb59363bff7d5a58e4794603c13"
	if got := buildIDRef(sha, sha); got != "ad2c6e7" {
		t.Errorf("buildIDRef(full SHA) = %q", got)
	}
	if got := buildIDRef("origin/main", sha); got != "main-ad2c6e7" {
		t.Errorf("buildIDRef(origin/main) = %q", got)
	}
}
