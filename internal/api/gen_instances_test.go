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

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/generation"
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

// Editing how an instance runs clears its VRAM measurement.
func TestGenEditInstanceClearsMeasurement(t *testing.T) {
	s := instanceServer(t)
	if err := s.gen.AddInstance(generation.Instance{ID: "m", Name: "m", ModelSetID: "ming-image", Port: 8081, Enabled: true, ReservedMiB: 20000, MeasuredPeakMiB: 18000}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleGenEditInstance(rec, instanceReq("PUT", "/", "m", "", url.Values{"model_set": {"ming-image"}, "gpu": {"0"}, "extra_args": {"--offload-to-cpu"}}))
	if rec.Code != http.StatusOK {
		t.Fatalf("edit: %d %s", rec.Code, rec.Body.String())
	}
	inst, _ := s.gen.Instance("m")
	if inst.ReservedMiB != 0 || inst.MeasuredPeakMiB != 0 || len(inst.ExtraArgs) != 1 {
		t.Errorf("after edit: %+v, want the measurement cleared and the argument stored", inst)
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
	if rec := act("disable"); rec.Code != http.StatusOK {
		t.Fatalf("disable: %d %s", rec.Code, rec.Body.String())
	}
	if inst, _ := s.gen.Instance("m"); inst.Enabled {
		t.Error("still enabled")
	}
	if rec := act("start"); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "disabled") {
		t.Errorf("start while disabled: %d %s", rec.Code, rec.Body.String())
	}
	if rec := act("enable"); rec.Code != http.StatusOK {
		t.Fatalf("enable: %d", rec.Code)
	}
	// The build's binary does not exist, so the start fails at launch;
	// the error is reported rather than swallowed.
	if rec := act("start"); rec.Code != http.StatusBadRequest {
		t.Errorf("start with a missing binary: %d, want 400", rec.Code)
	}
	if rec := act("dance"); rec.Code != http.StatusNotFound {
		t.Errorf("unknown action: %d", rec.Code)
	}

	rec := httptest.NewRecorder()
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
