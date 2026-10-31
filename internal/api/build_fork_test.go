package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/forks"
)

// forkBuildsServer is projectBuildsServer plus two Avalon builds: the
// pinned commit and a newer, untested one. The fork builds rank above the
// llama.cpp build, so a resolver that let them through would pick one.
func forkBuildsServer(t *testing.T) *Server {
	t.Helper()
	f, _ := forks.Find("avalon")
	pinned := f.PinnedSHA
	s := newTestServer(t)
	dir := s.cfg.DataDir
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	records := `[
	  {"id":"b9000-rocm","profile":"rocm","git_ref":"b9000","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","commit_count":9000},
	  {"id":"avalon-pinned-rocm","project":"avalon","profile":"rocm","git_ref":"` + pinned + `","git_sha":"` + pinned + `","status":"success","commit_count":11600},
	  {"id":"avalon-main-1111111-rocm","project":"avalon","profile":"rocm","git_ref":"origin/main","git_sha":"1111111111111111111111111111111111111111","status":"success","commit_count":11700}
	]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)
	return s
}

// The router and generation backends never run a fork build, even a newer
// one or one named outright.
func TestResolversSkipForkBuilds(t *testing.T) {
	s := forkBuildsServer(t)
	for _, id := range []string{"", "avalon-pinned-rocm"} {
		if got := s.resolveBuild(id); got == nil || got.ID != "b9000-rocm" {
			t.Errorf("resolveBuild(%q) = %v, want b9000-rocm", id, got)
		}
	}
	if got := s.resolveBackendBuild("fork", ""); got != nil {
		t.Errorf("resolveBackendBuild(fork) = %s", got.ID)
	}
}

func TestResolveForkBuild(t *testing.T) {
	s := forkBuildsServer(t)
	if got := s.resolveForkBuild("avalon", ""); got == nil || got.ID != "avalon-main-1111111-rocm" {
		t.Errorf("newest = %v", got)
	}
	if got := s.resolveForkBuild("avalon", "avalon-pinned-rocm"); got == nil || got.ID != "avalon-pinned-rocm" {
		t.Errorf("by ID = %v", got)
	}
	// Another project's build is not used even when named.
	if got := s.resolveForkBuild("avalon", "b9000-rocm"); got == nil || got.ID != "avalon-main-1111111-rocm" {
		t.Errorf("with a llama.cpp ID = %v", got)
	}
	if got := s.resolveForkBuild(builder.ProjectLlamaCpp, ""); got != nil {
		t.Errorf("llama.cpp is not a fork: %v", got)
	}
}

func TestJobsRefuseForkBuilds(t *testing.T) {
	s := forkBuildsServer(t)
	e := newJobEnv(s)
	if err := e.EnsureBuildActive(t.Context(), "avalon-pinned-rocm", false); err == nil || !strings.Contains(err.Error(), "fork") {
		t.Errorf("EnsureBuildActive(fork build) = %v", err)
	}
	if err := e.CheckBuildRunnable(t.Context(), "avalon-pinned-rocm"); err == nil || !strings.Contains(err.Error(), "fork") {
		t.Errorf("CheckBuildRunnable(fork build) = %v", err)
	}
	if err := refuseForkJobBuild(s.builder, "b9000-rocm"); err != nil {
		t.Errorf("llama.cpp build refused: %v", err)
	}
}

func TestBuildsListMarksUntestedForkBuilds(t *testing.T) {
	s := forkBuildsServer(t)
	req := httptest.NewRequest("GET", "/api/builds", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleListBuilds(rec, req)
	body := rec.Body.String()
	row := func(id string) string {
		i := strings.Index(body, id)
		if i < 0 {
			t.Fatalf("builds table lacks %s", id)
		}
		end := strings.Index(body[i:], "</tr>")
		return body[i : i+end]
	}
	if !strings.Contains(row("avalon-main-1111111-rocm"), "untested") {
		t.Error("latest-commit build not marked untested")
	}
	for _, id := range []string{"avalon-pinned-rocm", "b9000-rocm"} {
		if strings.Contains(row(id), "untested") {
			t.Errorf("%s marked untested", id)
		}
	}
}

func TestBuildsPageForkProjects(t *testing.T) {
	s := newTestServer(t)
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleBuildsPage(rec, httptest.NewRequest("GET", "/builds", nil))
		return rec.Body.String()
	}
	s.cfg.GenerationEnabled = true
	if strings.Contains(render(), `value="avalon"`) {
		t.Error("fork offered with forks off")
	}
	s.cfg.GenerationEnabled = false
	s.cfg.ForksEnabled = true
	page := render()
	for _, want := range []string{`id="build-project"`, `value="avalon"`, "Avalon (fork)", `data-backends="rocm,cuda"`} {
		if !strings.Contains(page, want) {
			t.Errorf("forks on: picker lacks %s", want)
		}
	}
	if strings.Contains(page, `value="stable-diffusion.cpp"`) {
		t.Error("generation project offered with generation off")
	}
}

func TestForkRefPicker(t *testing.T) {
	s := newTestServer(t)
	req := httptest.NewRequest("GET", "/api/builds/refs?project=avalon", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleListRefs(rec, req)
	body := rec.Body.String()
	f, _ := forks.Find("avalon")
	pinned := strings.Index(body, `value="`+f.PinnedSHA+`" selected`)
	latest := strings.Index(body, `value="latest"`)
	if pinned < 0 || latest < pinned || !strings.Contains(body, "Pinned (tested)") || !strings.Contains(body, "untested") {
		t.Errorf("fork ref picker:\n%s", body)
	}
	if strings.Count(body, "<option") != 2 {
		t.Errorf("fork ref picker has other refs:\n%s", body)
	}
}

func TestTriggerForkBuildGuards(t *testing.T) {
	s := newTestServer(t)
	post := func(ref string) *httptest.ResponseRecorder {
		form := url.Values{"project": {"avalon"}, "profile": {"rocm"}, "git_ref": {ref}}
		req := httptest.NewRequest("POST", "/api/builds", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		rec := httptest.NewRecorder()
		s.handleTriggerBuild(rec, req)
		return rec
	}
	if rec := post("latest"); rec.Code != http.StatusNotFound {
		t.Errorf("forks off: %d %s", rec.Code, rec.Body)
	}
	s.cfg.ForksEnabled = true
	for _, ref := range []string{"b9000", "origin/feature", "0123456789abcdef0123456789abcdef01234567"} {
		if rec := post(ref); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "pinned commit") {
			t.Errorf("ref %s: %d %s", ref, rec.Code, rec.Body)
		}
	}
}
