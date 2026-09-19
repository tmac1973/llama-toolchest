package api

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/autoconfig"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

func (s *Server) doAutoconfig(t *testing.T, method, path string, form url.Values) string {
	t.Helper()
	r := s.buildRouter()
	var body *strings.Reader
	if form != nil {
		body = strings.NewReader(form.Encode())
	} else {
		body = strings.NewReader("")
	}
	req := httptest.NewRequest(method, path, body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec.Body.String()
}

// finishedRun puts a completed autoconfigure result in place for the test
// model: context raised to 32K and a temperature set.
func finishedRun(t *testing.T, s *Server) {
	t.Helper()
	base, _ := s.registry.GetConfig(profTestID)
	proposed := *base
	proposed.ContextSize = 32768
	temp := 0.6
	proposed.Temperature = &temp
	s.autoconf.run = &autoconfigRun{modelID: profTestID, done: true, result: &autoconfig.Result{
		ModelID: profTestID, Base: *base, Proposed: proposed,
		Fit: models.FitResult{Fits: true, EstimateGiB: 9.5, BudgetGiB: 22},
		Notes: []models.ProfileNote{
			{Field: "context_size", Reason: "32,768 tokens, as requested.", Origin: "hardware fit"},
			{Field: "temperature", Reason: "From the card.", Origin: "model card"},
			{Reason: "The model card also mentions: use the chat template.", Origin: "model card"},
		},
	}}
}

func TestAutoconfigDialogWithoutHelper(t *testing.T) {
	s := newHelperServer(t)
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil)
	for _, want := range []string{"How long are your conversations", `value="medium" checked`, "No helper model is installed", "/settings"} {
		if !strings.Contains(out, want) {
			t.Errorf("dialog missing %q", want)
		}
	}
}

func TestAutoconfigRefusesASecondRun(t *testing.T) {
	s := newHelperServer(t)
	s.autoconf.run = &autoconfigRun{modelID: "other"}
	out := s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig", url.Values{"context_class": {"short"}})
	if !strings.Contains(out, "already running for other") {
		t.Errorf("second run not refused with a message:\n%s", out)
	}
	out = s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig", nil)
	if !strings.Contains(out, "already running for another model") || !strings.Contains(out, "disabled") {
		t.Errorf("dialog does not say another run is going:\n%s", out)
	}
}

func TestAutoconfigReviewShowsChangesWithReasons(t *testing.T) {
	s := newHelperServer(t)
	finishedRun(t, s)
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig/status", nil)
	for _, want := range []string{"Estimated <strong>9.5 GiB</strong> of 22.0 GiB", "Context size", "32,768 tokens, as requested.",
		"From the card.", "hardware fit", "Settings left as they are", "use the chat template", "Save and apply"} {
		if !strings.Contains(out, want) {
			t.Errorf("review missing %q", want)
		}
	}
	changed := out[strings.Index(out, "<tbody>"):strings.Index(out, "</tbody>")]
	if strings.Contains(changed, "Flash attention") {
		t.Error("an unchanged setting is listed among the changes")
	}
}

func TestAutoconfigSaveAsProfileLeavesLiveConfig(t *testing.T) {
	s := newHelperServer(t)
	finishedRun(t, s)
	out := s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/save", url.Values{"apply": {"0"}})
	if !strings.Contains(out, `Saved as profile &#34;Autoconfig&#34;`) && !strings.Contains(out, `Saved as profile "Autoconfig"`) {
		t.Errorf("no confirmation:\n%s", out)
	}
	p, err := s.registry.GetProfile(profTestID, "Autoconfig")
	if err != nil || p.Source != models.ProfileSourceAutoconfig || p.Config.ContextSize != 32768 || len(p.Notes) != 3 {
		t.Errorf("profile = %+v, %v", p, err)
	}
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.ContextSize == 32768 {
		t.Error("save without apply changed the live config")
	}
	if _, ok := s.autoconfigSnapshot(); ok {
		t.Error("the result was kept after saving")
	}
}

func TestAutoconfigSaveAndApply(t *testing.T) {
	s := newHelperServer(t)
	finishedRun(t, s)
	out := s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/save", url.Values{"apply": {"1"}})
	if !strings.Contains(out, "now in use") {
		t.Errorf("no apply confirmation:\n%s", out)
	}
	cfg, _ := s.registry.GetConfig(profTestID)
	if cfg.ContextSize != 32768 || cfg.ActiveProfile != "Autoconfig" {
		t.Errorf("live config = ctx %d, profile %q", cfg.ContextSize, cfg.ActiveProfile)
	}
}

func TestAutoconfigHintOnCardAndDismiss(t *testing.T) {
	s := newHelperServer(t)
	if err := s.registry.SetAutoconfigHint(profTestID, true); err != nil {
		t.Fatal(err)
	}
	m, _ := s.registry.Get(profTestID)
	card := renderModelCardPartial(t, modelCardData{Model: *m, CanAutoconfig: true})
	if !strings.Contains(card, "Autoconfigure can suggest settings") || !strings.Contains(card, `id="autoconfig-`) {
		t.Errorf("card lacks the hint or the container:\n%s", card)
	}
	s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/dismiss-hint", nil)
	if m, _ := s.registry.Get(profTestID); m.AutoconfigHint {
		t.Error("dismiss did not clear the hint")
	}
	if card := renderModelCardPartial(t, modelCardData{Model: *m, CanAutoconfig: false}); strings.Contains(card, ">\n                    Autoconfigure\n") {
		t.Error("Autoconfigure offered for a model that cannot use it")
	}
}

// A profile autoconfigure saved keeps its reasons, and the profile bar
// shows them while it is the active profile.
func TestProfileBarShowsAutoconfigReasons(t *testing.T) {
	s := newHelperServer(t)
	finishedRun(t, s)
	s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/save", url.Values{"apply": {"1"}})
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/config", nil)
	if !strings.Contains(out, "Why these settings") || !strings.Contains(out, "32,768 tokens, as requested.") {
		t.Errorf("profile bar lacks the reasons:\n%s", out)
	}
}
