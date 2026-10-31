package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
)

// instanceServer returns a test server with generation on, one complete sd
// set and one vulkan sd build (vulkan, so the GPU vendor check passes on
// any machine running the tests).
func instanceServer(t *testing.T) *Server {
	t.Helper()
	s := newTestServer(t)
	s.cfg.GenerationEnabled = true
	dir := s.cfg.DataDir
	store, err := generation.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	s.gen = store
	if err := store.AddModelSet(generation.ModelSet{ID: "ming-image", Name: "Ming-Image", Kind: generation.KindSD, Complete: true,
		Components: []generation.Component{{Slot: "diffusion_model", Path: "/m/dit.gguf"}}}); err != nil {
		t.Fatal(err)
	}
	if err := store.AddModelSet(generation.ModelSet{ID: "half", Name: "Half", Kind: generation.KindSD}); err != nil {
		t.Fatal(err)
	}
	os.MkdirAll(filepath.Join(dir, "config"), 0o755)
	records := `[{"id":"sd-master-1-abc-vulkan","project":"stable-diffusion.cpp","profile":"vulkan","git_ref":"master-1-abc","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","binary_path":"/nonexistent/sd-server","commit_count":1}]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(dir)
	return s
}

func instanceReq(method, target, id, action string, form url.Values) *http.Request {
	r := htmxReq(method, target, form)
	rc := chi.NewRouteContext()
	if id != "" {
		rc.URLParams.Add("id", id)
	}
	if action != "" {
		rc.URLParams.Add("action", action)
	}
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

func TestGenCreateInstanceDefaults(t *testing.T) {
	s := instanceServer(t)
	for i, want := range []string{"ming-image", "ming-image-2"} {
		rec := httptest.NewRecorder()
		s.handleGenCreateInstance(rec, instanceReq("POST", "/", "", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}}))
		if rec.Code != http.StatusOK {
			t.Fatalf("create %d: %d %s", i, rec.Code, rec.Body.String())
		}
		inst, err := s.gen.Instance(want)
		if err != nil {
			t.Fatalf("instance %s: %v", want, err)
		}
		if !inst.Enabled || inst.Port != 8081+i {
			t.Errorf("%s: enabled=%v port=%d, want enabled on %d", want, inst.Enabled, inst.Port, 8081+i)
		}
	}
}

func TestGenCreateInstanceRefusals(t *testing.T) {
	s := instanceServer(t)
	for name, form := range map[string]url.Values{
		"incomplete set": {"model_set": {"half"}, "gpu": {"0"}},
		"unknown set":    {"model_set": {"nope"}, "gpu": {"0"}},
		"denied arg":     {"model_set": {"ming-image"}, "gpu": {"0"}, "extra_args": {"--listen-port 9"}},
		"no gpu":         {"model_set": {"ming-image"}, "gpu": {"x"}},
	} {
		rec := httptest.NewRecorder()
		s.handleGenCreateInstance(rec, instanceReq("POST", "/", "", "", form))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%s: %d, want 400", name, rec.Code)
		}
	}

	// No build of the set's backend.
	s.builder = builder.NewBuilder(t.TempDir())
	rec := httptest.NewRecorder()
	s.handleGenCreateInstance(rec, instanceReq("POST", "/", "", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "Builds page") {
		t.Errorf("no build: %d %s", rec.Code, rec.Body.String())
	}
}

// A user can choose an instance's port in the generation port range, so
// clients keep working when an instance is made again.
func TestGenInstancePort(t *testing.T) {
	s := instanceServer(t)
	create := func(port string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleGenCreateInstance(rec, instanceReq("POST", "/", "", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "port": {port}}))
		return rec
	}
	if rec := create("8084"); rec.Code != http.StatusOK {
		t.Fatalf("create on 8084: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("ming-image"); inst.Port != 8084 {
		t.Errorf("port = %d, want the chosen 8084", inst.Port)
	}
	for port, want := range map[string]int{
		"8084": http.StatusConflict,   // used by ming-image
		"9000": http.StatusBadRequest, // outside 8081–8088
		"abc":  http.StatusBadRequest,
	} {
		if rec := create(port); rec.Code != want {
			t.Errorf("create on %s: %d, want %d", port, rec.Code, want)
		}
	}
	// Empty still picks the lowest free port.
	if rec := create(""); rec.Code != http.StatusOK {
		t.Fatalf("create with no port: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("ming-image-2"); inst.Port != 8081 {
		t.Errorf("automatic port = %d, want 8081", inst.Port)
	}

	edit := func(port string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleGenEditInstance(rec, instanceReq("PUT", "/", "ming-image-2", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "port": {port}}))
		return rec
	}
	if rec := edit("8084"); rec.Code != http.StatusConflict {
		t.Errorf("edit onto a used port: %d, want 409", rec.Code)
	}
	if rec := edit("8085"); rec.Code != http.StatusOK {
		t.Fatalf("edit to 8085: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("ming-image-2"); inst.Port != 8085 {
		t.Errorf("port after edit = %d, want 8085", inst.Port)
	}

	// An instance keeps its own port after the range moves away from it.
	s.cfg.GenPortStart = 9000
	for _, port := range []string{"8085", ""} {
		if rec := edit(port); rec.Code != http.StatusOK {
			t.Errorf("edit keeping port %q outside the new range: %d %s", port, rec.Code, rec.Body.String())
		}
	}
	if inst, _ := s.gen.Instance("ming-image-2"); inst.Port != 8085 {
		t.Errorf("port after edits = %d, want 8085 kept", inst.Port)
	}
}

// Editing how an instance runs clears its VRAM measurement.
func TestGenEditInstanceClearsMeasurement(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true, ReservedMiB: 20000, MeasuredPeakMiB: 18000}); err != nil {
		t.Fatal(err)
	}
	s.genMeasure.set("m", measureOutcome{Err: "a measurement of the old settings failed", At: time.Now()})
	rec := httptest.NewRecorder()
	s.handleGenEditInstance(rec, instanceReq("PUT", "/", "m", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "extra_args": {"--offload-to-cpu"}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("edit: %d %s", rec.Code, rec.Body.String())
	}
	inst, _ := s.gen.Instance("m")
	if inst.ReservedMiB != 0 || inst.MeasuredPeakMiB != 0 || len(inst.ExtraArgs) != 1 {
		t.Errorf("after edit: %+v, want the measurement cleared and the argument stored", inst)
	}
	if o := s.genMeasure.get("m"); o.Err != "" {
		t.Errorf("after edit the last measurement's error is still shown: %q", o.Err)
	}
}

func TestGenInstanceEnableDisableStart(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true}); err != nil {
		t.Fatal(err)
	}
	act := func(action string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleGenInstanceAction(rec, instanceReq("POST", "/", "m", action, nil))
		return rec
	}
	rec := act("disable")
	if rec.Code != http.StatusOK {
		t.Fatalf("disable: %d %s", rec.Code, rec.Body.String())
	}
	// A disabled instance's state reads "disabled", not "stopped".
	if body := rec.Body.String(); !strings.Contains(body, ">disabled</span>") || strings.Contains(body, ">stopped") {
		t.Errorf("state after disable:\n%s", body)
	}
	if inst, _ := s.gen.Instance("m"); inst.Enabled {
		t.Error("still enabled")
	}
	if rec := act("start"); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "disabled") {
		t.Errorf("start while disabled: %d %s", rec.Code, rec.Body.String())
	}
	rec = act("enable")
	if rec.Code != http.StatusOK {
		t.Fatalf("enable: %d", rec.Code)
	}
	if body := rec.Body.String(); !strings.Contains(body, ">stopped") || strings.Contains(body, ">disabled</span>") {
		t.Errorf("state after enable:\n%s", body)
	}
	// The build's binary does not exist, so the start fails at launch;
	// the error is reported rather than swallowed.
	if rec := act("start"); rec.Code != http.StatusBadRequest {
		t.Errorf("start with a missing binary: %d, want 400", rec.Code)
	}
	if rec := act("dance"); rec.Code != http.StatusNotFound {
		t.Errorf("unknown action: %d", rec.Code)
	}

	rec = httptest.NewRecorder()
	s.handleGenDeleteInstance(rec, instanceReq("DELETE", "/", "m", "", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("delete: %d", rec.Code)
	}
	if _, err := s.gen.Instance("m"); err == nil {
		t.Error("instance still stored after delete")
	}
}

func TestGenInstanceRoutesHiddenWhileOff(t *testing.T) {
	s := instanceServer(t)
	s.cfg.GenerationEnabled = false
	rec := httptest.NewRecorder()
	s.handleGenInstances(rec, instanceReq("GET", "/", "", "", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("list with the feature off: %d, want 404", rec.Code)
	}
}

func TestServerPageGenerationSection(t *testing.T) {
	s := instanceServer(t)
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleServerPage(rec, httptest.NewRequest("GET", "/", nil))
		return rec.Body.String()
	}
	if !strings.Contains(render(), `id="gen-instances"`) {
		t.Error("section missing while generation is on")
	}
	s.cfg.GenerationEnabled = false
	if strings.Contains(render(), `id="gen-instances"`) {
		t.Error("section shown while generation is off")
	}
}

// withAceBuild adds a vulkan acestep.cpp build next to the sd one.
func withAceBuild(t *testing.T, s *Server) {
	t.Helper()
	records := `[{"id":"sd-master-1-abc-vulkan","project":"stable-diffusion.cpp","profile":"vulkan","git_ref":"master-1-abc","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","binary_path":"/nonexistent/sd-server","commit_count":1},` +
		`{"id":"ace-master-1-abc-vulkan","project":"acestep.cpp","profile":"vulkan","git_ref":"master-1-abc","git_sha":"0123456789abcdef0123456789abcdef01234567","status":"success","binary_path":"/nonexistent/ace-server","commit_count":1}]`
	if err := os.WriteFile(filepath.Join(s.cfg.DataDir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	s.builder = builder.NewBuilder(s.cfg.DataDir)
}

// The Models page's New instance form is for one set: the set is fixed and
// only builds of its backend are offered.
func TestGenInstanceFormForSet(t *testing.T) {
	s := instanceServer(t)
	withAceBuild(t, s)
	form := func(target string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleGenInstanceForm(rec, instanceReq("GET", target, "", "", nil))
		return rec
	}
	rec := form("/api/gen/instances/form?set=ming-image")
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("form: %d %s", rec.Code, body)
	}
	if !strings.Contains(body, `type="hidden" name="model_set" value="ming-image"`) || strings.Contains(body, `<select name="model_set"`) {
		t.Errorf("the set is not fixed to ming-image:\n%s", body)
	}
	if !strings.Contains(body, "sd-master-1-abc-vulkan") || strings.Contains(body, "ace-master-1-abc-vulkan") {
		t.Errorf("want only the sd build offered:\n%s", body)
	}

	// Without a set, both backends' builds are offered, grouped.
	body = form("/api/gen/instances/form").Body.String()
	if !strings.Contains(body, `<optgroup label="stable-diffusion.cpp">`) || !strings.Contains(body, "ace-master-1-abc-vulkan") {
		t.Errorf("want both backends' builds:\n%s", body)
	}

	if rec := form("/api/gen/instances/form?set=nope"); rec.Code != http.StatusNotFound {
		t.Errorf("unknown set: %d, want 404", rec.Code)
	}
	if body := form("/api/gen/instances/form?set=half").Body.String(); !strings.Contains(body, "the download is not finished") {
		t.Errorf("incomplete set: want it explained:\n%s", body)
	}
}

// A build of another backend is refused, not saved and quietly replaced by
// the newest build of the right one.
func TestGenCreateInstanceRefusesWrongBuild(t *testing.T) {
	s := instanceServer(t)
	withAceBuild(t, s)
	rec := httptest.NewRecorder()
	s.handleGenCreateInstance(rec, instanceReq("POST", "/", "", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "build": {"ace-master-1-abc-vulkan"}}))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "not a successful stable-diffusion.cpp build") {
		t.Errorf("wrong build: %d %s", rec.Code, rec.Body.String())
	}
	if len(s.gen.Instances()) != 0 {
		t.Errorf("an instance was saved: %+v", s.gen.Instances())
	}
}

// The Models page greys out New instance for a set that has an instance,
// and for one that is not fully downloaded.
func TestGenModelSetsNewInstanceButton(t *testing.T) {
	s := instanceServer(t)
	s.downloader = huggingface.NewDownloader(s.cfg.DataDir, filepath.Join(s.cfg.DataDir, "models"), "")
	render := func() string {
		rec := httptest.NewRecorder()
		s.handleGenModels(rec, htmxReq("GET", "/api/gen/models", nil))
		return rec.Body.String()
	}
	if body := render(); !strings.Contains(body, `hx-get="/api/gen/instances/form?set=ming-image"`) || strings.Contains(body, `form?set=half`) {
		t.Errorf("want the button for ming-image only:\n%s", body)
	}
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true}); err != nil {
		t.Fatal(err)
	}
	if body := render(); strings.Contains(body, `form?set=ming-image`) || !strings.Contains(body, "already has an instance") {
		t.Errorf("want the button greyed out once ming-image has an instance:\n%s", body)
	}
}
