package api

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// projectBuildsServer returns a test server whose builder holds one
// successful build of each project.
func projectBuildsServer(t *testing.T) *Server {
	t.Helper()
	s := newTestServer(t)
	dir := s.cfg.DataDir
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	records := `[
	  {"id":"b9000-rocm","profile":"rocm","git_ref":"b9000","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","commit_count":9000},
	  {"id":"sd-master-1000-abc-rocm","project":"stable-diffusion.cpp","profile":"rocm","git_ref":"master-1000-abc","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","commit_count":99999,"has_web_ui":true},
	  {"id":"ace-master-rocm","project":"acestep.cpp","profile":"rocm","git_ref":"master","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","commit_count":300}
	]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)
	return s
}

// The router must never run an sd or ace build, even when one is named
// explicitly as the active build.
func TestResolveBuildIgnoresOtherProjects(t *testing.T) {
	s := projectBuildsServer(t)
	if got := s.resolveBuild("sd-master-1000-abc-rocm"); got == nil || got.ID != "b9000-rocm" {
		t.Errorf("resolveBuild(sd build) = %v, want the llama.cpp build", got)
	}
	if got := s.resolveBuild(""); got == nil || got.ID != "b9000-rocm" {
		t.Errorf("resolveBuild(\"\") = %v, want b9000-rocm", got)
	}
}

func TestResolveBackendBuild(t *testing.T) {
	s := projectBuildsServer(t)
	if got := s.resolveBackendBuild("sd", ""); got == nil || got.ID != "sd-master-1000-abc-rocm" {
		t.Errorf("sd = %v", got)
	}
	if got := s.resolveBackendBuild("ace", "ace-master-rocm"); got == nil || got.ID != "ace-master-rocm" {
		t.Errorf("ace by ID = %v", got)
	}
	// A build of the wrong project is not used even when named.
	if got := s.resolveBackendBuild("sd", "b9000-rocm"); got == nil || got.ID != "sd-master-1000-abc-rocm" {
		t.Errorf("sd with a llama.cpp ID = %v, want the newest sd build", got)
	}
	if got := s.resolveBackendBuild("whisper", ""); got != nil {
		t.Errorf("unknown kind = %v, want nil", got)
	}
}

func TestBuildsListShowsAllProjects(t *testing.T) {
	s := projectBuildsServer(t)
	req := httptest.NewRequest("GET", "/api/builds", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleListBuilds(rec, req)
	for _, id := range []string{"b9000-rocm", "sd-master-1000-abc-rocm", "ace-master-rocm"} {
		if !strings.Contains(rec.Body.String(), id) {
			t.Errorf("builds table lacks %s", id)
		}
	}
}

// The flag preview applies the project's own switches, as the build does.
func TestProfileOptionsPreviewPerProject(t *testing.T) {
	s := newTestServer(t)
	get := func(query string) string {
		req := httptest.NewRequest("GET", "/api/builds/options?"+query, nil)
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleProfileOptions(rec, req)
		return rec.Body.String()
	}

	sd := get("project=stable-diffusion.cpp&profile=cuda")
	if !strings.Contains(sd, "-DSD_CUDA=ON") || strings.Contains(sd, "LLAMA_OPENSSL") {
		t.Errorf("sd.cpp cuda preview lacks SD_CUDA or still has LLAMA_OPENSSL:\n%s", sd)
	}
	if !strings.Contains(sd, "SD_SERVER_BUILD_FRONTEND") {
		t.Error("sd.cpp preview lacks the web UI switch")
	}
	llama := get("profile=cuda")
	if strings.Contains(llama, "SD_CUDA") || !strings.Contains(llama, "LLAMA_OPENSSL") {
		t.Errorf("llama.cpp preview changed:\n%s", llama)
	}
	if cpu := get("project=acestep.cpp&profile=cpu"); !strings.Contains(cpu, "can only be built with the rocm, cuda, vulkan profiles") {
		t.Errorf("acestep.cpp cpu preview lacks the profile warning:\n%s", cpu)
	}
}

func TestBuildsPageProjectPickerOnlyWhenEnabled(t *testing.T) {
	s := newTestServer(t)
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleBuildsPage(rec, httptest.NewRequest("GET", "/builds", nil))
		return rec.Body.String()
	}
	if strings.Contains(render(), `id="build-project"`) {
		t.Error("project picker shown while generation backends are off")
	}
	s.cfg.GenerationEnabled = true
	on := render()
	for _, want := range []string{`id="build-project"`, `value="stable-diffusion.cpp"`, `data-backends="rocm,cuda,vulkan"`, `value="acestep.cpp"`} {
		if !strings.Contains(on, want) {
			t.Errorf("project picker lacks %s", want)
		}
	}
}
