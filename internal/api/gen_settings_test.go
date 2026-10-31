package api

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

func putSettingsForm(s *Server, form url.Values) *httptest.ResponseRecorder {
	req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	return rec
}

func TestGenerationSwitchSavesAndReloads(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})

	putSettingsForm(s, url.Values{"generation_touched": {"1"}, "generation_enabled": {"on"}})
	if !s.cfg.GenerationEnabled {
		t.Fatal("GenerationEnabled not set by the switch")
	}
	cfg, err := config.Load(s.configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.GenerationEnabled {
		t.Error("GenerationEnabled not persisted to the config file")
	}

	// An unchecked switch posts only the marker.
	putSettingsForm(s, url.Values{"generation_touched": {"1"}})
	if s.cfg.GenerationEnabled {
		t.Error("GenerationEnabled still on after turning the switch off")
	}
}

// Switching generation on in Settings starts the auto-start instances,
// as starting llama-toolchest does. It used to start nothing, so they
// stayed stopped until the next restart. Only the switch from off to on
// starts them: a save while it is already on, or turning it off, does not.
func TestGenerationSwitchOnStartsAutoStartInstances(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	started := make(chan struct{}, 4)
	s.genAutoStartFn = func() { started <- struct{}{} }
	expect := func(what string, want bool) {
		t.Helper()
		select {
		case <-started:
			if !want {
				t.Errorf("%s: auto-start ran", what)
			}
		case <-time.After(200 * time.Millisecond):
			if want {
				t.Errorf("%s: auto-start did not run", what)
			}
		}
	}

	on := url.Values{"generation_touched": {"1"}, "generation_enabled": {"on"}}
	putSettingsForm(s, on)
	expect("switched on", true)
	putSettingsForm(s, on)
	expect("saved while on", false)
	putSettingsForm(s, url.Values{"generation_touched": {"1"}})
	expect("switched off", false)
	putSettingsForm(s, on)
	expect("switched on again", true)
}

// The generation settings (evict, prompt cache) are drawn only while
// generation is on, so turning it on or off reloads the Settings page.
// Before, the evict switch stayed hidden until a manual reload. Other
// saves do not reload: they would lose the "Settings saved." line.
func TestGenerationSwitchReloadsTheSettingsPage(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	s.genAutoStartFn = func() {}
	put := func(form url.Values) string {
		req := httptest.NewRequest("PUT", "/api/settings", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("HX-Request", "true")
		rec := httptest.NewRecorder()
		s.handleUpdateSettings(rec, req)
		return rec.Header().Get("HX-Refresh")
	}
	on := url.Values{"generation_touched": {"1"}, "generation_enabled": {"on"}}
	off := url.Values{"generation_touched": {"1"}}
	for _, c := range []struct {
		what string
		form url.Values
		want string
	}{
		{"switched on", on, "true"},
		{"saved while on", on, ""},
		{"another setting", url.Values{"auto_start_touched": {"1"}}, ""},
		{"switched off", off, "true"},
	} {
		if got := put(c.form); got != c.want {
			t.Errorf("%s: HX-Refresh = %q, want %q", c.what, got, c.want)
		}
	}
}

func TestGenPortRangeValidated(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})

	rec := putSettingsForm(s, url.Values{"gen_port_start": {"8075"}, "gen_port_count": {"8"}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "llama-server port 8080") {
		t.Errorf("overlapping range: %d %q, want 400 naming the llama-server port", rec.Code, rec.Body.String())
	}
	if s.cfg.GenPortStart != 0 {
		t.Errorf("rejected range was stored: %d", s.cfg.GenPortStart)
	}

	rec = putSettingsForm(s, url.Values{"gen_port_start": {"9100"}, "gen_port_count": {"4"}})
	if rec.Code != http.StatusOK {
		t.Fatalf("valid range: %d %s", rec.Code, rec.Body.String())
	}
	if start, count := s.cfg.GenPorts(false); start != 9100 || count != 4 {
		t.Errorf("range = %d,%d, want 9100,4", start, count)
	}
}

func TestGenPortRangeFixedInContainer(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	s.inContainerFn = func() bool { return true }

	rec := putSettingsForm(s, url.Values{"gen_port_start": {"9100"}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "inside a container") {
		t.Errorf("container port change: %d %q, want 400", rec.Code, rec.Body.String())
	}
}

// Saving unrelated settings must not add generation keys to the file of an
// install that never turned the feature on.
func TestUnrelatedSaveAddsNoGenerationKeys(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	putSettingsForm(s, url.Values{"auto_start_touched": {"1"}, "auto_start": {"on"}})

	saved, err := os.ReadFile(s.configPath)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(saved), "gen") {
		t.Errorf("config file gained generation keys:\n%s", saved)
	}
}

func TestSettingsPageGenerationSection(t *testing.T) {
	base := testTemplates(t, "templates/layout.html", "templates/settings.html", "templates/partials/*.html")
	published := false
	render := func(enabled, inContainer bool) string {
		data := struct {
			pageData
			ProxyEndpoint      string
			LlamaPort          int
			HasAPIKey          bool
			HasHFToken         bool
			HasMSToken         bool
			DefaultSource      string
			HasExtURL          bool
			ExternalURL        string
			ExternalURLEnv     string
			ExternalURLInUse   string
			DataDir            string
			ModelsDir          string
			DefaultModelsDir   string
			AutoStart          bool
			RuntimeEnvOpts     []config.RuntimeEnvOption
			RuntimeEnv         map[string]string
			RuntimeEnvExtra    string
			EnvBackends        []string
			ActiveBackend      string
			EnvWarnings        []string
			EffectiveEnv       []envLine
			GenerationEnabled  bool
			ForksEnabled       bool
			GenEvict           bool
			GenKeepPromptCache bool
			GenPortStart       int
			GenPortEnd         int
			GenPortCount       int
			InContainer        bool
			GenPortsPublished  bool
		}{GenerationEnabled: enabled, GenPortStart: 8081, GenPortEnd: 8088, GenPortCount: 8, InContainer: inContainer,
			GenPortsPublished: !inContainer || published}
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "content", data); err != nil {
			t.Fatalf("execute settings page: %v", err)
		}
		return buf.String()
	}

	// switchTag returns the generation switch's own <input ...> tag.
	switchTag := func(page string) string {
		i := strings.Index(page, `name="generation_enabled"`)
		if i < 0 {
			t.Fatal("generation switch missing from the page")
		}
		return page[i : i+strings.IndexByte(page[i:], '>')]
	}

	off := render(false, false)
	if strings.Contains(switchTag(off), "checked") {
		t.Error("switch checked while the feature is off")
	}
	if !strings.Contains(off, "8081–8088") {
		t.Error("port range not shown")
	}
	if strings.Contains(off, "readonly") {
		t.Error("port fields read-only outside a container")
	}
	if on := render(true, false); !strings.Contains(switchTag(on), "checked") {
		t.Error("switch not checked while the feature is on")
	}
	if c := render(false, true); !strings.Contains(c, "readonly") || !strings.Contains(c, "setup.sh") {
		t.Error("container view should make the ports read-only and point at setup.sh")
	}

	// The note about unpublished web UI ports shows only for a turned-on
	// feature in a container that does not publish them.
	const notPublished = "Web UI ports are not published"
	if strings.Contains(render(true, false), notPublished) {
		t.Error("host install told its ports are not published")
	}
	if strings.Contains(render(false, true), notPublished) {
		t.Error("ports note shown while the feature is off")
	}
	if !strings.Contains(render(true, true), notPublished) {
		t.Error("container without published ports not told")
	}
	published = true
	if strings.Contains(render(true, true), notPublished) {
		t.Error("ports note shown although setup.sh published them")
	}
}
