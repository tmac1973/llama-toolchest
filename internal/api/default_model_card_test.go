package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/forks"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func renderDefaultCard(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleDefaultModelCard(rec, httptest.NewRequest(http.MethodGet, "/api/settings/default-model", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("card: %d %s", rec.Code, rec.Body)
	}
	return rec.Body.String()
}

func putDefaultForm(t *testing.T, s *Server, on bool, name, pin string) *httptest.ResponseRecorder {
	t.Helper()
	v := url.Values{"default_model_form": {"1"}, "default_model_name": {name}, "default_model_pin": {pin}}
	if on {
		v.Set("default_model_on", "on")
	}
	return putMarginForm(t, s, v)
}

// The card's "Now" line says where requests go and why, in each case.
func TestDefaultModelCardNow(t *testing.T) {
	s, f, a := startFakeRouter(t)
	f.reset(map[string]string{}, nil)
	b := addChatModel(t, s, "beta", models.ModelConfig{Enabled: true}, false)

	if card := renderDefaultCard(t, s); !strings.Contains(card, "Now: no model.") {
		t.Errorf("no target:\n%s", card)
	}
	s.defaultState.note(a.ID, false, time.Now())
	if card := renderDefaultCard(t, s); !strings.Contains(card, a.PublicName()+"</kbd> (used last; loads on the next request)") {
		t.Errorf("last used:\n%s", card)
	}
	f.reset(map[string]string{b.ID: "loaded"}, nil)
	if card := renderDefaultCard(t, s); !strings.Contains(card, b.PublicName()+"</kbd> (loaded, used last)") {
		t.Errorf("loaded:\n%s", card)
	}
	s.cfg.DefaultModel.Pin = a.ID
	card := renderDefaultCard(t, s)
	if !strings.Contains(card, a.PublicName()+"</kbd> (pinned)") {
		t.Errorf("pinned:\n%s", card)
	}
	if !strings.Contains(card, `<option value="`+a.ID+`" selected>`) {
		t.Errorf("the pin is not selected:\n%s", card)
	}
	s.cfg.DefaultModel.Off = true
	if card := renderDefaultCard(t, s); !strings.Contains(card, "Now: off.") || strings.Contains(card, `name="default_model_on" checked`) {
		t.Errorf("off:\n%s", card)
	}
	s.cfg.DefaultModel.Off = false
	s.registry.SetConfig(b.ID, &models.ModelConfig{Enabled: true, Aliases: []string{"default"}})
	if card := renderDefaultCard(t, s); !strings.Contains(card, "The name is off: the model") || !strings.Contains(card, "Now: off (a model has the name).") {
		t.Errorf("clash:\n%s", card)
	}
}

// The pin options are the chat models, then the fork instances, each by
// label; a stale pin is cleared when the card loads; a fork pin stays,
// and is still offered, while forks are off.
func TestDefaultModelCardOptions(t *testing.T) {
	s, _ := forkInstanceServer(t)
	addChatModel(t, s, "zeta", models.ModelConfig{Enabled: true}, false)
	addChatModel(t, s, "alpha", models.ModelConfig{Enabled: true}, false)
	addChatModel(t, s, "off", models.ModelConfig{Enabled: false}, false)
	addChatModel(t, s, "embedder", models.ModelConfig{Enabled: true}, true)
	s.forks.AddInstance(forks.Instance{ID: "ik", ForkID: "avalon", GPU: 0, Port: 1})

	var labels []string
	for _, o := range s.defaultModelInfo().Options {
		labels = append(labels, o.Label)
	}
	got := strings.Join(labels, ",")
	alpha, _ := s.registry.Get("alpha")
	zeta, _ := s.registry.Get("zeta")
	if want := alpha.PublicName() + "," + zeta.PublicName() + ",ik (fork)"; got != want {
		t.Errorf("options = %s, want %s", got, want)
	}

	s.cfg.DefaultModel.Pin = "off"
	renderDefaultCard(t, s)
	if s.cfg.DefaultModel.Pin != "" {
		t.Error("a pin to a disabled model survived the card load")
	}

	s.cfg.DefaultModel.Pin = config.DefaultModelForkPrefix + "ik"
	s.cfg.ForksEnabled = false
	card := renderDefaultCard(t, s)
	if s.cfg.DefaultModel.Pin == "" || !strings.Contains(card, `<option value="fork:ik" selected>ik (fork, forks are off)</option>`) {
		t.Errorf("forks off: pin %q, card:\n%s", s.cfg.DefaultModel.Pin, card)
	}
	// Saving another field sends the kept pin back, and it is accepted.
	on, name, pin := false, "", "fork:ik"
	s.cfgMu.Lock()
	err := s.applyDefaultModelLocked(defaultModelUpdate{On: &on, Name: &name, Pin: &pin})
	s.cfgMu.Unlock()
	if err != nil {
		t.Errorf("save with the kept fork pin: %v", err)
	}
}

func TestDefaultModelSave(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	p := addChatModel(t, s, "pinned", models.ModelConfig{Enabled: true}, false)
	addChatModel(t, s, "aliased", models.ModelConfig{Enabled: true, Aliases: []string{"taken"}}, false)

	rec := putDefaultForm(t, s, true, " chat ", p.ID)
	if rec.Code != http.StatusOK || rec.Header().Get("HX-Trigger") != "settingsSaved" {
		t.Fatalf("save: %d %s %v", rec.Code, rec.Body, rec.Header())
	}
	if want := (config.DefaultModelConfig{Name: "chat", Pin: p.ID}); s.cfg.DefaultModel != want {
		t.Errorf("saved %+v, want %+v", s.cfg.DefaultModel, want)
	}

	for _, tc := range []struct{ name, pin, want string }{
		{"two words", p.ID, "use 1 to 64 letters"},
		{"taken", p.ID, `"taken": the model`},
		{"chat", "nobody", "not a chat model or fork instance"},
	} {
		rec := putDefaultForm(t, s, false, tc.name, tc.pin)
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), tc.want) {
			t.Errorf("%q %q: %d %s", tc.name, tc.pin, rec.Code, rec.Body)
		}
		if s.cfg.DefaultModel.Off {
			t.Errorf("%q %q: a refused save switched the name off", tc.name, tc.pin)
		}
	}

	// Blank is the default name; off is saved as off.
	putDefaultForm(t, s, false, "  ", "")
	if want := (config.DefaultModelConfig{Off: true}); s.cfg.DefaultModel != want {
		t.Errorf("saved %+v, want %+v", s.cfg.DefaultModel, want)
	}

	// The same through JSON, and back in the settings answer.
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"default_model_on":true,"default_model_name":"chat","default_model_pin":"`+p.ID+`"}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	var got settingsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("%d %s", rec.Code, rec.Body)
	}
	if !got.DefaultModelOn || got.DefaultModelName != "chat" || got.DefaultModelPin != p.ID {
		t.Errorf("JSON save answered %+v", got)
	}
	req = httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"default_model_name":"taken"}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("JSON clash: %d %s", rec.Code, rec.Body)
	}

	// A form without the card's marker leaves the name alone.
	putMarginForm(t, s, url.Values{"auto_start_touched": {"1"}})
	if s.cfg.DefaultModel.Name != "chat" || s.cfg.DefaultModel.Off {
		t.Errorf("another card's save changed the name: %+v", s.cfg.DefaultModel)
	}
}

// The target gets a "default" badge on its model card, and a fork
// instance target on its row.
func TestDefaultModelBadges(t *testing.T) {
	s, _ := forkInstanceServer(t)
	p := addChatModel(t, s, "pinned", models.ModelConfig{Enabled: true}, false)
	other := addChatModel(t, s, "other", models.ModelConfig{Enabled: true}, false)
	s.forks.AddInstance(forks.Instance{ID: "ik", ForkID: "avalon", GPU: 0, Port: 1})
	s.cfg.DefaultModel.Pin = p.ID

	card := func(m *models.Model) string {
		rec := httptest.NewRecorder()
		id, _ := s.defaultTargetIDs()
		s.renderModelCard(rec, m, nil, false, false, id)
		return rec.Body.String()
	}
	badge := `title="Requests to &#34;default&#34; go to this model.">default</mark>`
	if !strings.Contains(card(p), badge) {
		t.Errorf("the pinned model has no badge:\n%s", card(p))
	}
	if strings.Contains(card(other), ">default</mark>") {
		t.Error("another model has the badge")
	}
	forkBadge := func() bool {
		for _, r := range s.forkRows(s.forks) {
			if r.ID == "ik" {
				return r.DefaultBadge != ""
			}
		}
		return false
	}
	if forkBadge() {
		t.Error("the fork row has the badge while a model is pinned")
	}
	s.cfg.DefaultModel.Pin = config.DefaultModelForkPrefix + "ik"
	if !forkBadge() || strings.Contains(card(p), ">default</mark>") {
		t.Error("the badge did not move to the pinned fork instance")
	}
	s.cfg.DefaultModel.Off = true
	if forkBadge() {
		t.Error("the badge shows while the name is off")
	}
}
