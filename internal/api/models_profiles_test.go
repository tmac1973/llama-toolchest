package api

import (
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
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

const profTestID = "org--r-GGUF--m-Q4_K_M"

// newProfileServer builds the parts of a Server the config panel and the
// profile handlers use, with one installed model.
func newProfileServer(t *testing.T, activeBuild string) *Server {
	t.Helper()
	dir := t.TempDir()
	reg := models.NewRegistry(dir, filepath.Join(dir, "models"))
	if err := reg.Add(&models.Model{ID: profTestID, ModelID: "org/r-GGUF", Filename: "m-Q4_K_M.gguf", Quant: "Q4_K_M"}); err != nil {
		t.Fatal(err)
	}
	s := &Server{
		cfg:         &config.Config{DataDir: dir, ActiveBuild: activeBuild},
		registry:    reg,
		builder:     builder.NewBuilder(dir),
		monitor:     monitor.New(time.Hour),
		process:     process.NewManager(),
		dirtyModels: map[string]bool{},
	}
	s.pages = s.parseTemplates()
	return s
}

// doProfileRequest sends an htmx request through a router holding the config and
// profile routes, and returns the response body.
func (s *Server) doProfileRequest(t *testing.T, method, path string, form url.Values) (int, string) {
	t.Helper()
	r := chi.NewRouter()
	r.Get("/api/models/{id}/config", s.handleGetModelConfig)
	r.Put("/api/models/{id}/config", s.handleUpdateModelConfig)
	r.Post("/api/models/{id}/profiles", s.handleSaveProfile)
	r.Post("/api/models/{id}/profiles/apply", s.handleApplyProfile)
	r.Post("/api/models/{id}/profiles/delete", s.handleDeleteProfile)

	req := httptest.NewRequest(method, path, strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec.Code, rec.Body.String()
}

func setLiveContext(t *testing.T, s *Server, ctx int) {
	t.Helper()
	cfg, _ := s.registry.GetConfig(profTestID)
	next := *cfg
	next.ContextSize = ctx
	if err := s.registry.SetConfig(profTestID, &next); err != nil {
		t.Fatal(err)
	}
}

func TestProfileHandlersSaveRestoreDelete(t *testing.T) {
	s := newProfileServer(t, "b100")
	base := "/api/models/" + profTestID
	setLiveContext(t, s, 16384)

	_, body := s.doProfileRequest(t, "POST", base+"/profiles", url.Values{"profile_name": {"Long context"}})
	if !strings.Contains(body, "Saved the current settings as profile") || !strings.Contains(body, "Matches profile <strong>Long context</strong>") {
		t.Fatalf("save response missing confirmation or status:\n%s", body)
	}

	setLiveContext(t, s, 4096)
	_, body = s.doProfileRequest(t, "GET", base+"/config", nil)
	if !strings.Contains(body, "changed since") {
		t.Errorf("panel does not report the edit since the profile")
	}

	_, body = s.doProfileRequest(t, "POST", base+"/profiles/apply", url.Values{"profile": {"Long context"}})
	if !strings.Contains(body, "Restored profile") {
		t.Fatalf("apply response missing confirmation:\n%s", body)
	}
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.ContextSize != 16384 {
		t.Errorf("context after restore = %d, want 16384", cfg.ContextSize)
	}

	_, body = s.doProfileRequest(t, "POST", base+"/profiles/delete", url.Values{"profile": {"Long context"}})
	if !strings.Contains(body, "Deleted profile") || !strings.Contains(body, "No saved profiles yet") {
		t.Errorf("delete response wrong:\n%s", body)
	}
}

func TestProfileSaveWithoutNameShowsError(t *testing.T) {
	s := newProfileServer(t, "")
	code, body := s.doProfileRequest(t, "POST", "/api/models/"+profTestID+"/profiles", url.Values{"profile_name": {"  "}})
	if code != http.StatusOK || !strings.Contains(body, "Type a name for the profile first.") {
		t.Errorf("code %d, body lacks the name error:\n%s", code, body)
	}
	if !strings.Contains(body, `<form id="model-config-form"`) {
		t.Error("an error banner replaced the form instead of sitting above it")
	}
}

// The config form's autosave posts only its own fields: it must not touch
// profiles, and it returns the form plus an out-of-band status line.
func TestAutosaveLeavesProfilesAndRefreshesStatus(t *testing.T) {
	s := newProfileServer(t, "")
	if _, err := s.registry.SaveProfile(profTestID, "Keep", models.ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	form := url.Values{"enabled": {"true"}, "gpu_layers": {"999"}, "context_size": {"2048"}, "threads": {"8"}}
	code, body := s.doProfileRequest(t, "PUT", "/api/models/"+profTestID+"/config", form)
	if code != http.StatusOK {
		t.Fatalf("autosave code %d: %s", code, body)
	}
	if n := len(s.registry.Profiles(profTestID)); n != 1 {
		t.Errorf("profiles after autosave = %d, want 1", n)
	}
	if p, _ := s.registry.GetProfile(profTestID, "Keep"); p.Config.ContextSize == 2048 {
		t.Error("autosave changed the saved profile")
	}
	if strings.Contains(body, `class="profile-bar"`) {
		t.Error("autosave response re-renders the whole profile bar")
	}
	if !strings.Contains(body, `hx-swap-oob="true"`) || !strings.Contains(body, "changed since") {
		t.Errorf("autosave response lacks the out-of-band status update:\n%s", body)
	}
}

func TestProfileRestoreRefusesMissingDraftModel(t *testing.T) {
	s := newProfileServer(t, "")
	cfg, _ := s.registry.GetConfig(profTestID)
	next := *cfg
	next.SpecType = "draft"
	next.DraftModelPath = filepath.Join(t.TempDir(), "gone.gguf")
	s.registry.SetConfig(profTestID, &next)
	if _, err := s.registry.SaveProfile(profTestID, "With draft", models.ProfileSourceUser, ""); err != nil {
		t.Fatal(err)
	}
	next.SpecType, next.DraftModelPath = "", ""
	s.registry.SetConfig(profTestID, &next)

	_, body := s.doProfileRequest(t, "POST", "/api/models/"+profTestID+"/profiles/apply", url.Values{"profile": {"With draft"}})
	if !strings.Contains(body, "was not restored") || !strings.Contains(body, "no longer on disk") {
		t.Errorf("restore of a profile with a missing draft model was not refused:\n%s", body)
	}
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.SpecType != "" {
		t.Error("refused restore still changed the live config")
	}
}

func TestProfileRestoreWarnsAboutOtherBuild(t *testing.T) {
	s := newProfileServer(t, "b200")
	if _, err := s.registry.SaveProfile(profTestID, "Old", models.ProfileSourceUser, "b100"); err != nil {
		t.Fatal(err)
	}
	_, body := s.doProfileRequest(t, "POST", "/api/models/"+profTestID+"/profiles/apply", url.Values{"profile": {"Old"}})
	if !strings.Contains(body, "saved with build b100 and the active build is b200") {
		t.Errorf("no build warning:\n%s", body)
	}
}

func TestProfileBarDisabledWhenReadOnly(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "config", "models.json"), []byte(`{"schema_version": 99,
		"models": {"`+profTestID+`": {"id": "`+profTestID+`", "model_id": "org/r-GGUF", "filename": "m-Q4_K_M.gguf"}},
		"configs": {"`+profTestID+`": {"enabled": true}}}`), 0o644)
	s := newProfileServer(t, "")
	s.registry = models.NewRegistry(dir, filepath.Join(dir, "models"))

	_, body := s.doProfileRequest(t, "POST", "/api/models/"+profTestID+"/profiles", url.Values{"profile_name": {"x"}})
	if !strings.Contains(body, "Profile not saved") || !strings.Contains(body, "newer version") {
		t.Errorf("read-only save did not explain itself:\n%s", body)
	}
	bar := body[strings.Index(body, `class="profile-bar"`):]
	bar = bar[:strings.Index(bar, `class="profile-status"`)]
	if strings.Count(bar, "disabled") < 5 {
		t.Errorf("profile controls are not all disabled on a read-only registry")
	}
}

// The bar must sit outside the autosaving form, or browsing the picker
// would save the config.
func TestProfileBarIsOutsideTheForm(t *testing.T) {
	s := newProfileServer(t, "")
	s.registry.SaveProfile(profTestID, "A", models.ProfileSourceUser, "")
	_, body := s.doProfileRequest(t, "GET", "/api/models/"+profTestID+"/config", nil)
	barAt := strings.Index(body, `class="profile-bar"`)
	formAt := strings.Index(body, `<form id="model-config-form"`)
	if barAt < 0 || formAt < 0 || barAt > formAt {
		t.Fatalf("profile bar (at %d) must come before the form (at %d)", barAt, formAt)
	}
	form := body[formAt:strings.Index(body, "</form>")]
	for _, name := range []string{`name="profile"`, `name="profile_name"`} {
		if strings.Contains(form, name) {
			t.Errorf("%s is inside the autosaving form", name)
		}
	}
}
