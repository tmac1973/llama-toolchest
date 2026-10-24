package builder

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestFindProject(t *testing.T) {
	for _, id := range []string{"", ProjectLlamaCpp} {
		if p, ok := FindProject(id); !ok || p.ID != ProjectLlamaCpp || p.IDPrefix != "" {
			t.Errorf("FindProject(%q) = %+v, %v; want llama.cpp with no ID prefix", id, p.ID, ok)
		}
	}
	if p, ok := FindProject(ProjectSDCpp); !ok || p.Kind != "sd" || p.Required.Name != "sd-server" {
		t.Errorf("stable-diffusion.cpp = %+v, %v", p, ok)
	}
	if p, ok := FindProject(ProjectAceStep); !ok || p.Kind != "ace" || p.Required.Name != "ace-server" {
		t.Errorf("acestep.cpp = %+v, %v", p, ok)
	}
	if _, ok := FindProject("whisper.cpp"); ok {
		t.Error("unknown project found")
	}
}

func TestProjectBackends(t *testing.T) {
	llama, _ := FindProject(ProjectLlamaCpp)
	sd, _ := FindProject(ProjectSDCpp)
	for _, b := range []string{"rocm", "cuda", "vulkan", "metal", "cpu"} {
		if !llama.SupportsBackend(b) {
			t.Errorf("llama.cpp refuses %s", b)
		}
	}
	for b, want := range map[string]bool{"rocm": true, "cuda": true, "vulkan": true, "metal": false, "cpu": false} {
		if got := sd.SupportsBackend(b); got != want {
			t.Errorf("sd.cpp SupportsBackend(%s) = %v, want %v", b, got, want)
		}
	}
}

// sd.cpp's own switches are added for the profile's GGML_* ones, and the
// llama.cpp-only OpenSSL switch is dropped.
func TestSDMapFlags(t *testing.T) {
	prof, _ := FindProfile("rocm")
	flags := copyFlags(prof.CMakeFlags)
	flags["LLAMA_OPENSSL"] = "ON"
	sd, _ := FindProject(ProjectSDCpp)
	sd.MapFlags(flags)
	if flags["SD_HIPBLAS"] != "ON" || flags["GGML_HIP"] != "ON" {
		t.Errorf("flags = %v, want SD_HIPBLAS=ON next to GGML_HIP=ON", flags)
	}
	if _, ok := flags["GPU_TARGETS"]; !ok {
		t.Error("GPU_TARGETS dropped")
	}
	if _, ok := flags["LLAMA_OPENSSL"]; ok {
		t.Error("LLAMA_OPENSSL kept for sd.cpp")
	}

	cuda := map[string]string{"GGML_CUDA": "ON"}
	sd.MapFlags(cuda)
	if cuda["SD_CUDA"] != "ON" || cuda["SD_HIPBLAS"] != "" {
		t.Errorf("cuda flags = %v", cuda)
	}
}

func TestProjectOptions(t *testing.T) {
	has := func(opts []BuildOption, flag string) bool {
		return slices.ContainsFunc(opts, func(o BuildOption) bool { return o.Flag == flag })
	}
	llama := ProjectOptions(ProjectLlamaCpp, "rocm")
	if !has(llama, "LLAMA_OPENSSL") || has(llama, sdFrontendFlag) {
		t.Error("llama.cpp options should be the profile's own")
	}
	sd := ProjectOptions(ProjectSDCpp, "rocm")
	if has(sd, "LLAMA_OPENSSL") || !has(sd, sdFrontendFlag) || !has(sd, "GGML_NATIVE") {
		t.Errorf("sd.cpp options = %v; want GGML_* kept, LLAMA_* dropped, the web UI switch added", sd)
	}
	if has(ProjectOptions(ProjectAceStep, "cuda"), sdFrontendFlag) {
		t.Error("acestep.cpp got sd.cpp's web UI switch")
	}
}

func TestNotesURL(t *testing.T) {
	sd, _ := FindProject(ProjectSDCpp)
	if got := sd.NotesURL("master-1000-95ff762"); got != "https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1000-95ff762" {
		t.Errorf("NotesURL = %q", got)
	}
	if got := sd.NotesURL("latest"); got != sd.LatestNotesURL {
		t.Errorf("latest NotesURL = %q", got)
	}
}

// Hash-only and workflow tags cannot be ordered and are left out; the
// numbered ones sort by their number, not as text.
func TestSortSDTags(t *testing.T) {
	got := sortSDTags([]string{"master-ff4fdbb", "master-999-aaaaaaa", "workflow-302-fad6178", "master-1000-95ff762", "master-99-bbbbbbb", ""})
	want := []string{"master-1000-95ff762", "master-999-aaaaaaa", "master-99-bbbbbbb"}
	if !slices.Equal(got, want) {
		t.Errorf("sortSDTags = %v, want %v", got, want)
	}
}

func TestFindBuilt(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "ace-server"), []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	ace, _ := FindProject(ProjectAceStep)
	if got := findBuilt(dir, ace.Required); got != filepath.Join(dir, "ace-server") {
		t.Errorf("findBuilt = %q, want the top-level ace-server", got)
	}
	sd, _ := FindProject(ProjectSDCpp)
	if got := findBuilt(dir, sd.Required); got != "" {
		t.Errorf("findBuilt found %q for a missing sd-server", got)
	}
}

// The router, benchmarks and autotune use List and SuccessfulBuildsRanked;
// a generation backend's build must never reach them. Records from before
// projects existed are llama.cpp builds.
func TestBuildListsSeparateProjects(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	records := `[
	  {"id":"b9000-rocm","profile":"rocm","git_ref":"b9000","status":"success","commit_count":9000},
	  {"id":"sd-master-1000-abc-rocm","project":"stable-diffusion.cpp","profile":"rocm","git_ref":"master-1000-abc","status":"success","commit_count":99999},
	  {"id":"ace-master-rocm","project":"acestep.cpp","profile":"rocm","git_ref":"master","status":"success","commit_count":300}
	]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	b := NewBuilder(dir)

	if got := b.List(); len(got) != 1 || got[0].ID != "b9000-rocm" {
		t.Errorf("List = %v, want only the llama.cpp build", got)
	}
	if got := b.LatestSuccessfulBuild(); got == nil || got.ID != "b9000-rocm" {
		t.Errorf("LatestSuccessfulBuild = %v, want b9000-rocm even though sd's count is higher", got)
	}
	if got := b.SuccessfulBuildsRankedFor(ProjectSDCpp); len(got) != 1 || got[0].ID != "sd-master-1000-abc-rocm" {
		t.Errorf("sd builds = %v", got)
	}
	if got := len(b.ListAll()); got != 3 {
		t.Errorf("ListAll has %d builds, want 3", got)
	}
	if r, _ := b.Find("b9000-rocm"); r.ProjectID() != ProjectLlamaCpp {
		t.Errorf("a record without a project = %q, want llama.cpp", r.ProjectID())
	}
}

// Refused before anything is cloned, so these need no network.
func TestBuildProjectRefusals(t *testing.T) {
	b := NewBuilder(t.TempDir())
	_, err := b.BuildProject(context.Background(), ProjectSDCpp, "cpu", "latest", "", false, nil, "", "")
	if err == nil || !strings.Contains(err.Error(), "rocm, cuda, vulkan") {
		t.Errorf("sd.cpp cpu build: %v, want a profile refusal", err)
	}
	if _, err := b.BuildProject(context.Background(), "whisper.cpp", "rocm", "latest", "", false, nil, "", ""); err == nil {
		t.Error("unknown project accepted")
	}
}

func TestBuildIDRef(t *testing.T) {
	for _, tc := range []struct{ ref, sha, want string }{
		{"b11868", "abc", "b11868"},
		{"master-1000-95ff762", "95ff762aaaa", "master-1000-95ff762"},
		{"origin/master", "de858d8aaaa", "master-de858d8"},
		{"origin/feature/x", "0123456789", "feature-x-0123456"},
		{"pr/123", "", "pr-123"},
	} {
		if got := buildIDRef(tc.ref, tc.sha); got != tc.want {
			t.Errorf("buildIDRef(%q, %q) = %q, want %q", tc.ref, tc.sha, got, tc.want)
		}
	}
}
