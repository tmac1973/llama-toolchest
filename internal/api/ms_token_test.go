package api

import (
	"bytes"
	"context"
	"html/template"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/modelscope"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/presets"
	"github.com/tmac1973/llama-toolchest/web"
)

// renderSettings renders the settings page with the two token flags set
// as given. The struct mirrors the handler's anonymous one; a mismatch
// shows up here as a template execution error.
func renderSettings(t *testing.T, hasHF, hasMS bool) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/settings.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	data := struct {
		pageData
		ProxyEndpoint    string
		LlamaPort        int
		HasAPIKey        bool
		HasHFToken       bool
		HasMSToken       bool
		DefaultSource    string
		HasExtURL        bool
		ExternalURL      string
		DataDir          string
		ModelsDir        string
		DefaultModelsDir string
		AutoStart        bool
		RuntimeEnvOpts   []config.RuntimeEnvOption
		RuntimeEnv       map[string]string
		RuntimeEnvExtra  string
		EnvBackends      []string
		ActiveBackend    string
		EnvWarnings      []string
		EffectiveEnv     []envLine
	}{HasHFToken: hasHF, HasMSToken: hasMS}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "content", data); err != nil {
		t.Fatalf("execute settings page: %v", err)
	}
	return buf.String()
}

// newSettingsServer builds the minimum Server that handleUpdateSettings
// needs: a config, somewhere to persist it, and the live clients whose
// credentials a save now pushes into.
func newSettingsServer(t *testing.T, cfg *config.Config) *Server {
	t.Helper()
	dir := t.TempDir()
	cfg.DataDir = dir
	s := &Server{
		cfg:        cfg,
		configPath: filepath.Join(dir, "llama-toolchest.yaml"),
		hfClient:   huggingface.NewClient(cfg.HFToken),
		msClient:   modelscope.NewClient(cfg.MSToken),
		presets:    presets.NewFetcher(filepath.Join(dir, "cache"), cfg.HFToken),
		downloader: huggingface.NewDownloader(dir, dir, cfg.HFToken),
	}
	s.downloader.RegisterProvider(modelsource.SourceModelScope, huggingface.Provider{
		URL:   s.msClient.DownloadURL,
		Token: cfg.MSToken,
	})
	return s
}

func TestSettingsPageHasBothTokenFields(t *testing.T) {
	out := renderSettings(t, false, false)
	for _, want := range []string{`name="hf_token"`, `name="ms_token"`} {
		if !strings.Contains(out, want) {
			t.Errorf("settings page missing %s", want)
		}
	}
}

// A set token is never echoed back into the page — only its presence is,
// via the placeholder. Rendering the value would leak it to anyone who
// can view the page or a screenshot of it.
func TestSettingsPagePlaceholdersReportPresenceOnly(t *testing.T) {
	set := renderSettings(t, true, true)
	if strings.Count(set, "(token is set)") != 2 {
		t.Errorf("both fields should report a set token; got %d occurrences", strings.Count(set, "(token is set)"))
	}
	unset := renderSettings(t, false, false)
	if strings.Contains(unset, "(token is set)") {
		t.Error("unset tokens should not claim to be set")
	}
	if !strings.Contains(unset, "ms_...") {
		t.Error("unset ModelScope field should show its placeholder hint")
	}
}

// Saving one token must not blank the other: the form posts both fields
// and an empty one means "leave it alone", not "clear it". Getting this
// wrong would wipe a working HuggingFace token the first time someone
// saved a ModelScope one.
func TestUpdateSettingsLeavesTheOtherTokenAlone(t *testing.T) {
	s := newSettingsServer(t, &config.Config{HFToken: "hf_existing", MSToken: "ms_existing"})

	form := url.Values{"ms_token": {"ms_new"}, "hf_token": {""}}
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	s.handleUpdateSettings(httptest.NewRecorder(), req)

	if s.cfg.MSToken != "ms_new" {
		t.Errorf("MSToken = %q, want ms_new", s.cfg.MSToken)
	}
	if s.cfg.HFToken != "hf_existing" {
		t.Errorf("HFToken = %q, want the existing value kept", s.cfg.HFToken)
	}
}

// The JSON path treats a present key as authoritative, including an
// explicit empty string, which is how a token gets cleared.
func TestUpdateSettingsJSONSetsAndClears(t *testing.T) {
	s := newSettingsServer(t, &config.Config{MSToken: "ms_existing"})
	put := func(body string) {
		req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		s.handleUpdateSettings(httptest.NewRecorder(), req)
	}

	put(`{"ms_token":"ms_new"}`)
	if s.cfg.MSToken != "ms_new" {
		t.Errorf("MSToken = %q, want ms_new", s.cfg.MSToken)
	}
	put(`{"ms_token":""}`)
	if s.cfg.MSToken != "" {
		t.Errorf("MSToken = %q, want it cleared by an explicit empty value", s.cfg.MSToken)
	}
	// An absent key leaves the value alone.
	s.cfg.MSToken = "ms_kept"
	put(`{"llama_port":9999}`)
	if s.cfg.MSToken != "ms_kept" {
		t.Errorf("MSToken = %q, want it untouched when the key is absent", s.cfg.MSToken)
	}
}

// The token must reach the config file, or it is forgotten on restart.
func TestUpdateSettingsPersistsMSToken(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	path := s.configPath

	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(`{"ms_token":"ms_persisted"}`))
	req.Header.Set("Content-Type", "application/json")
	s.handleUpdateSettings(httptest.NewRecorder(), req)

	saved, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("config not written: %v", err)
	}
	if !strings.Contains(string(saved), "ms_persisted") {
		t.Errorf("ms_token missing from the saved config:\n%s", saved)
	}
}

func TestSettingsPageDefaultSourceSelect(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/settings.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	render := func(def string) string {
		data := struct {
			pageData
			ProxyEndpoint    string
			LlamaPort        int
			HasAPIKey        bool
			HasHFToken       bool
			HasMSToken       bool
			DefaultSource    string
			HasExtURL        bool
			ExternalURL      string
			DataDir          string
			ModelsDir        string
			DefaultModelsDir string
			AutoStart        bool
			RuntimeEnvOpts   []config.RuntimeEnvOption
			RuntimeEnv       map[string]string
			RuntimeEnvExtra  string
			EnvBackends      []string
			ActiveBackend    string
			EnvWarnings      []string
			EffectiveEnv     []envLine
		}{DefaultSource: def}
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "content", data); err != nil {
			t.Fatalf("execute (default=%q): %v", def, err)
		}
		return buf.String()
	}

	// The selected option must reflect the stored preference, or saving
	// the form would silently reset it to HuggingFace.
	ms := render("modelscope")
	if !strings.Contains(ms, `value="modelscope" selected`) {
		t.Error("ModelScope preference not preselected on the settings page")
	}
	if strings.Contains(ms, `value="hf" selected`) {
		t.Error("both options marked selected")
	}
	hf := render("")
	if !strings.Contains(hf, `value="hf" selected`) {
		t.Error("unset preference should preselect HuggingFace")
	}
	if !strings.Contains(hf, `name="default_model_source"`) {
		t.Error("settings page missing the default source control")
	}
}

// The preference round-trips through the settings API, and an
// unrecognized value is normalized rather than stored.
func TestUpdateSettingsDefaultSource(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	put := func(body string) {
		req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		s.handleUpdateSettings(httptest.NewRecorder(), req)
	}

	put(`{"default_model_source":"modelscope"}`)
	if s.cfg.DefaultModelSource != "modelscope" {
		t.Errorf("DefaultModelSource = %q, want modelscope", s.cfg.DefaultModelSource)
	}
	put(`{"default_model_source":"nonsense"}`)
	if s.cfg.DefaultModelSource != "hf" {
		t.Errorf("DefaultModelSource = %q, want an unrecognized value normalized to hf", s.cfg.DefaultModelSource)
	}

	// Form path: a select always posts, so presence means chosen.
	form := url.Values{"default_model_source": {"modelscope"}}
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	s.handleUpdateSettings(httptest.NewRecorder(), req)
	if s.cfg.DefaultModelSource != "modelscope" {
		t.Errorf("form path: DefaultModelSource = %q, want modelscope", s.cfg.DefaultModelSource)
	}
}

// The point of applying credentials on save: a token set in Settings must
// reach the next outbound request, not the next restart. Assert on the
// Authorization header a client actually sends, since that is the only
// thing that decides whether a gated repo downloads.
func TestSavedTokenReachesTheWire(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[]}}}`))
	}))
	defer srv.Close()

	s := newSettingsServer(t, &config.Config{})
	ms := modelscope.NewClient("")
	ms.SetHTTPClientForTest(srv.Client(), srv.URL)
	s.msClient = ms

	// Before: no credential.
	if _, err := ms.Search(context.Background(), "q"); err != nil {
		t.Fatalf("search: %v", err)
	}
	if gotAuth != "" {
		t.Fatalf("unexpected Authorization before a token was set: %q", gotAuth)
	}

	// Save a token through the real handler.
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(`{"ms_token":"ms_live"}`))
	req.Header.Set("Content-Type", "application/json")
	s.handleUpdateSettings(httptest.NewRecorder(), req)

	// After: the very next request carries it, with no restart.
	if _, err := ms.Search(context.Background(), "q"); err != nil {
		t.Fatalf("search after save: %v", err)
	}
	if gotAuth != "Bearer ms_live" {
		t.Errorf("Authorization = %q, want the token saved a moment ago", gotAuth)
	}
}

// Swapping a token while requests are in flight must not race: that is
// the reason the token lives behind a lock inside each client rather than
// the Server swapping client pointers. Run with -race to mean anything.
func TestTokenSwapIsSafeUnderConcurrency(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"Code":200,"Success":true,"Data":{"Model":{"Models":[]}}}`))
	}))
	defer srv.Close()

	c := modelscope.NewClient("start")
	c.SetHTTPClientForTest(srv.Client(), srv.URL)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				c.Search(context.Background(), "q")
			}
		}()
	}
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				c.SetToken("tok")
			}
		}(i)
	}
	wg.Wait()
}
