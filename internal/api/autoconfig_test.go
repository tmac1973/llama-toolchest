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
// renderCard renders the test model's card the way the models page does.
func (s *Server) renderCard(t *testing.T) string {
	t.Helper()
	m, err := s.registry.Get(profTestID)
	if err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.renderModelCard(rec, m, nil, false, false)
	return rec.Body.String()
}

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

// The suggestion is shown on every model that has not been through
// Autoconfigure, and stops once the model has an Autoconfig profile or
// the user dismisses it.
func TestAutoconfigHintShownUntilUsedOrDismissed(t *testing.T) {
	s := newHelperServer(t)
	show := func() bool {
		return strings.Contains(s.renderCard(t), "Autoconfigure can suggest settings")
	}
	if !show() {
		t.Error("a model that has never been autoconfigured shows no suggestion")
	}
	s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/dismiss-hint", nil)
	if m, _ := s.registry.Get(profTestID); !m.AutoconfigDismissed {
		t.Error("dismiss was not recorded")
	}
	if show() {
		t.Error("the suggestion survived being dismissed")
	}

	// A model with an Autoconfig profile does not need the suggestion.
	if err := s.registry.SetAutoconfigDismissed(profTestID, false); err != nil {
		t.Fatal(err)
	}
	finishedRun(t, s)
	s.doAutoconfig(t, "POST", "/api/models/"+profTestID+"/autoconfig/save", url.Values{"apply": {"0"}})
	if show() {
		t.Error("the suggestion is still shown after an Autoconfig profile was saved")
	}
}

// The card offers the button and a place to render into.
func TestModelCardOffersAutoconfigure(t *testing.T) {
	s := newHelperServer(t)
	out := s.renderCard(t)
	if !strings.Contains(out, "Autoconfigure\n") || !strings.Contains(out, `id="autoconfig-`) {
		t.Errorf("card lacks the button or the container:\n%s", out)
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

// A preset that sets six sampling values carries the same sentence on
// each of them; a row must not repeat it.
func TestReviewRowDoesNotRepeatTheSameReason(t *testing.T) {
	s := newHelperServer(t)
	base, _ := s.registry.GetConfig(profTestID)
	proposed := *base
	temp := 1.0
	proposed.Temperature = &temp
	proposed.SamplingPreset = "unsloth"
	same := "From the publisher's \"Unsloth recommended\" sampling settings."
	s.autoconf.run = &autoconfigRun{modelID: profTestID, done: true, result: &autoconfig.Result{
		ModelID: profTestID, Base: *base, Proposed: proposed,
		Notes: []models.ProfileNote{
			{Field: "temperature", Reason: same, Origin: "publisher preset"},
			{Field: "sampling_preset", Reason: same, Origin: "publisher preset"},
		},
	}}
	out := s.doAutoconfig(t, "GET", "/api/models/"+profTestID+"/autoconfig/status", nil)
	if n := strings.Count(out, "Unsloth recommended"); n != 1 {
		t.Errorf("the same reason appears %d times, want 1:\n%s", n, out)
	}
}

func TestGPUAssignText(t *testing.T) {
	for in, want := range map[string]string{
		"": "all GPUs", "all": "all GPUs", "0": "GPU 0", "0-1": "GPUs 0-1", "0,2": "GPUs 0,2",
		"tensor-2": "tensor parallelism over GPUs 2", "custom": "a custom split",
	} {
		if got := gpuAssignText(in); got != want {
			t.Errorf("gpuAssignText(%q) = %q, want %q", in, got, want)
		}
	}
}
