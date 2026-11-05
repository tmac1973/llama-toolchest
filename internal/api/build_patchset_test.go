package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/patchsets"
)

// patchBuildsServer has an upstream build, a patched build of the tested
// rdna-boosts release, and one of a newer, untested release.
func patchBuildsServer(t *testing.T) *Server {
	t.Helper()
	set, _ := patchsets.Find("rdna-boosts")
	s := newTestServer(t)
	dir := s.cfg.DataDir
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	records := `[
	  {"id":"b11880-rocm","profile":"rocm","git_ref":"b11880","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","commit_count":11880},
	  {"id":"rdna-b11887-r206-rocm","profile":"rocm","git_ref":"` + set.Pinned.Tag + `","git_sha":"efaa60db0553f0a8f25c89cf62c573bddc22b911","status":"success","commit_count":11887,
	   "patch_set":"rdna-boosts","patch_release":"` + set.Pinned.Tag + `","patch_base":"` + set.Pinned.Base + `"},
	  {"id":"rdna-b11887-r999-rocm","profile":"rocm","git_ref":"v16-6050b467c-r999","git_sha":"1111111111111111111111111111111111111111","status":"success","commit_count":11887,
	   "patch_set":"rdna-boosts","patch_release":"v16-6050b467c-r999","patch_base":"` + set.Pinned.Base + `"}
	]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)
	return s
}

func TestPatchSetChoicesNeedTheirBackend(t *testing.T) {
	if got := patchSetChoices([]builder.Backend{{Name: "cpu", Available: true}, {Name: "rocm"}}); len(got) != 0 {
		t.Errorf("offered without ROCm: %+v", got)
	}
	got := patchSetChoices([]builder.Backend{{Name: "rocm", Available: true}})
	if len(got) != 1 || got[0].ID != "patchset:rdna-boosts" || got[0].Label != "llama.cpp + RDNA boosts" || got[0].Backends != "rocm" {
		t.Fatalf("choices = %+v", got)
	}
	if !strings.Contains(got[0].PatchNote, "6050b467c") || !strings.Contains(got[0].PatchNote, "16 patches") {
		t.Errorf("note = %q", got[0].PatchNote)
	}
	// The newest release can have another base and patch count.
	if n := got[0].PatchLatestNote; n == "" || strings.Contains(n, "6050b467c") || strings.Contains(n, "16 patches") {
		t.Errorf("latest note = %q", n)
	}
}

func TestBuildsPageOffersPatchSetWithROCm(t *testing.T) {
	withBackends(t, builder.Backend{Name: "rocm", Available: true}, builder.Backend{Name: "cpu", Available: true})
	s := newTestServer(t)
	rec := httptest.NewRecorder()
	s.handleBuildsPage(rec, httptest.NewRequest("GET", "/builds", nil))
	page := rec.Body.String()
	for _, want := range []string{`id="build-project"`, `value="patchset:rdna-boosts"`, "llama.cpp &#43; RDNA boosts", `data-patch-note="Upstream llama.cpp at 6050b467c`} {
		if !strings.Contains(page, want) {
			t.Errorf("page lacks %s", want)
		}
	}
	// llama.cpp first, the patch set right after it.
	if strings.Index(page, `value="llama.cpp"`) > strings.Index(page, `value="patchset:rdna-boosts"`) {
		t.Error("patch set listed before llama.cpp")
	}
}

func TestPatchSetRefPicker(t *testing.T) {
	s := newTestServer(t)
	set, _ := patchsets.Find("rdna-boosts")
	req := httptest.NewRequest("GET", "/api/builds/refs?project=patchset:rdna-boosts&refresh=1", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleListRefs(rec, req)
	body := rec.Body.String()
	pinned := strings.Index(body, `value="`+set.Pinned.Tag+`" selected`)
	latest := strings.Index(body, `value="latest"`)
	if pinned < 0 || latest < pinned || !strings.Contains(body, "r206 (tested)") || !strings.Contains(body, "Newest (untested)") {
		t.Errorf("patch set ref picker:\n%s", body)
	}
	if strings.Count(body, "<option") != 2 {
		t.Errorf("patch set ref picker has other refs:\n%s", body)
	}
}

func TestPatchSetBuildOptionsAreLlamaCpps(t *testing.T) {
	s := newTestServer(t)
	rec := httptest.NewRecorder()
	s.handleProfileOptions(rec, httptest.NewRequest("GET", "/api/builds/options?project=patchset:rdna-boosts&profile=rocm", nil))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "GGML_HIP_RCCL") {
		t.Errorf("options: %d\n%s", rec.Code, rec.Body)
	}
}

func TestTriggerPatchSetBuildGuards(t *testing.T) {
	s := newTestServer(t)
	post := func(project, profile, ref string) *httptest.ResponseRecorder {
		form := url.Values{"project": {project}, "profile": {profile}, "git_ref": {ref}}
		req := httptest.NewRequest("POST", "/api/builds", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		rec := httptest.NewRecorder()
		s.handleTriggerBuild(rec, req)
		return rec
	}
	for _, ref := range []string{"b11887", "v16-6050b467c-r1", "6050b467c07a95c7b87debd4ddac2f8ba8d9b807"} {
		if rec := post("patchset:rdna-boosts", "rocm", ref); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "tested release") {
			t.Errorf("ref %s: %d %s", ref, rec.Code, rec.Body)
		}
	}
	if rec := post("patchset:nope", "rocm", ""); rec.Code != http.StatusNotFound {
		t.Errorf("unknown patch set: %d %s", rec.Code, rec.Body)
	}
	// A valid choice reaches the builder, which refuses the CPU profile
	// before it fetches anything.
	if rec := post("patchset:rdna-boosts", "cpu", ""); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "rocm profiles") {
		t.Errorf("cpu profile: %d %s", rec.Code, rec.Body)
	}
}

func TestBuildsListShowsPatchedBuilds(t *testing.T) {
	s := patchBuildsServer(t)
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
	if r := row("rdna-b11887-r206-rocm"); !strings.Contains(r, "RDNA boosts r206") || strings.Contains(r, "untested") || !strings.Contains(r, "Base: 6050b46 (b11887)") {
		t.Errorf("tested patched row: %s", r)
	}
	if r := row("rdna-b11887-r999-rocm"); !strings.Contains(r, "RDNA boosts r999") || !strings.Contains(r, "untested") || !strings.Contains(r, "not the tested release") {
		t.Errorf("untested patched row: %s", r)
	}
	if r := row("b11880-rocm"); strings.Contains(r, "RDNA") || strings.Contains(r, "untested") {
		t.Errorf("upstream row: %s", r)
	}
}

func TestBuildInfoShowsPatchSet(t *testing.T) {
	s := patchBuildsServer(t)
	info := func(id string) string {
		req := httptest.NewRequest("GET", "/api/builds/"+id+"/info", nil)
		req.Header.Set("HX-Request", "true")
		rctx := chi.NewRouteContext()
		rctx.URLParams.Add("id", id)
		req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))
		rec := httptest.NewRecorder()
		s.handleBuildInfo(rec, req)
		return rec.Body.String()
	}
	body := info("rdna-b11887-r206-rocm")
	for _, want := range []string{"Patch set", "RDNA boosts r206", "(tested)", "releases/tag/v16-6050b467c-r206", "<code>6050b46</code> · b11887"} {
		if !strings.Contains(body, want) {
			t.Errorf("info lacks %q:\n%s", want, body)
		}
	}
	if body := info("b11880-rocm"); strings.Contains(body, "Patch set") {
		t.Errorf("upstream info shows a patch set:\n%s", body)
	}
}

// "Auto" passes over patched builds while an upstream build can run, even
// one that ranks lower; with only patched builds it takes one. A saved
// choice is honoured.
func TestResolveBuildAutoSkipsPatchedBuilds(t *testing.T) {
	s := patchBuildsServer(t)
	if got := s.resolveBuild(""); got == nil || got.ID != "b11880-rocm" {
		t.Errorf("Auto = %v, want b11880-rocm", got)
	}
	if got := s.resolveBuild("rdna-b11887-r206-rocm"); got == nil || got.ID != "rdna-b11887-r206-rocm" {
		t.Errorf("saved patched build = %v", got)
	}
	if err := s.builder.Delete("b11880-rocm"); err != nil {
		t.Fatal(err)
	}
	if got := s.resolveBuild(""); got == nil || got.PatchSet != "rdna-boosts" {
		t.Errorf("Auto with only patched builds = %v", got)
	}
}

// A patched build gets the launch options of upstream at its base.
func TestPatchedBuildTargetIsItsBase(t *testing.T) {
	s := patchBuildsServer(t)
	b, _ := s.builder.Find("rdna-b11887-r206-rocm")
	got := buildTarget(b)
	want := buildTarget(&builder.BuildResult{Profile: "rocm", CommitCount: 11887})
	if !reflect.DeepEqual(got, want) {
		t.Errorf("target = %+v, want %+v", got, want)
	}
	if got.Version != 11887 || got.RouterMemoryOption() || !got.DraftUBatchOption() {
		t.Errorf("target %+v: router memory %v, draft ubatch %v", got, got.RouterMemoryOption(), got.DraftUBatchOption())
	}
}

func TestJobsAcceptPatchedBuilds(t *testing.T) {
	s := patchBuildsServer(t)
	if err := refuseForkJobBuild(s.builder, "rdna-b11887-r206-rocm"); err != nil {
		t.Errorf("patched build refused as a job build: %v", err)
	}
	// List feeds the job form and the router; patched builds are in it.
	found := false
	for _, b := range s.builder.List() {
		found = found || b.ID == "rdna-b11887-r206-rocm"
	}
	if !found {
		t.Error("patched build not in the llama.cpp builds")
	}
}

func TestBuildPickersLabelPatchedBuilds(t *testing.T) {
	data := jobFormData(t)
	data.Builds = []buildOpt{{ID: "b1", Profile: "rocm"}, {ID: "rdna-b11887-r206-rocm", Profile: "rocm", PatchLabel: "RDNA boosts r206"}}
	out := renderJobFormPartial(t, data)
	if !strings.Contains(out, ", RDNA boosts r206)") || strings.Count(out, "RDNA boosts") != 1 {
		t.Errorf("job form build labels:\n%s", out)
	}

	s := patchBuildsServer(t)
	choices, _ := s.serverBuildChoices("")
	labels := map[string]string{}
	for _, c := range choices {
		labels[c.ID] = c.PatchLabel
	}
	if labels["rdna-b11887-r206-rocm"] != "RDNA boosts r206" || labels["rdna-b11887-r999-rocm"] != "RDNA boosts r999" || labels["b11880-rocm"] != "" {
		t.Errorf("server picker labels = %v", labels)
	}
}
