package api

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// serve sends one request through the full router and returns the
// response.
func serve(s *Server, req *http.Request) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(rec, req)
	return rec
}

// addV1Model registers a model with the given config.
func addV1Model(t *testing.T, s *Server, id, repo string, cfg models.ModelConfig) *models.Model {
	t.Helper()
	m := &models.Model{
		ID: id, ModelID: repo, Quant: "Q4_K_M", Filename: id + ".gguf",
		FilePath: filepath.Join(t.TempDir(), id+".gguf"),
	}
	if err := s.registry.Add(m); err != nil {
		t.Fatal(err)
	}
	if err := s.registry.SetConfig(id, &cfg); err != nil {
		t.Fatal(err)
	}
	return m
}

// /v1/models is what OpenAI clients read to pick a model. It lists only
// enabled models that are not the app's own helper, under their public
// name with their aliases.
func TestV1ModelsListsEnabledServingModels(t *testing.T) {
	s := newTestServer(t)
	on := addV1Model(t, s, "on", "org/On-GGUF", models.ModelConfig{Enabled: true, Aliases: []string{"chat"}})
	addV1Model(t, s, "off", "org/Off-GGUF", models.ModelConfig{Enabled: false})
	addV1Model(t, s, "helper", "org/Helper-GGUF", models.ModelConfig{Enabled: true})
	if err := s.registry.SetHelperRole("helper", true); err != nil {
		t.Fatal(err)
	}

	rec := serve(s, httptest.NewRequest("GET", "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("HTTP %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string   `json:"id"`
			Aliases []string `json:"aliases"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body)
	}
	if out.Object != "list" || len(out.Data) != 1 {
		t.Fatalf("list = %+v, want only the enabled serving model", out)
	}
	if out.Data[0].ID != on.PublicName() || len(out.Data[0].Aliases) != 1 || out.Data[0].Aliases[0] != "chat" {
		t.Errorf("entry = %+v, want id %q with alias chat", out.Data[0], on.PublicName())
	}
}

// A client may name a model by registry ID, public name or alias; all
// three must find the same model, and an unknown name gets an
// OpenAI-shaped 404 so client libraries show a useful message.
func TestV1ModelFindsByAnyName(t *testing.T) {
	s := newTestServer(t)
	m := addV1Model(t, s, "reg-id", "org/Model-GGUF", models.ModelConfig{Enabled: true, Aliases: []string{"chat"}})

	for _, name := range []string{"reg-id", m.PublicName(), "chat"} {
		rec := serve(s, httptest.NewRequest("GET", "/v1/models/"+name, nil))
		if rec.Code != http.StatusOK {
			t.Errorf("%s: HTTP %d: %s", name, rec.Code, rec.Body)
			continue
		}
		var got struct {
			ID string `json:"id"`
		}
		json.Unmarshal(rec.Body.Bytes(), &got)
		if got.ID != m.PublicName() {
			t.Errorf("%s: id = %q, want %q", name, got.ID, m.PublicName())
		}
	}

	rec := serve(s, httptest.NewRequest("GET", "/v1/models/nobody", nil))
	if rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), `"model_not_found"`) {
		t.Errorf("unknown model: HTTP %d: %s", rec.Code, rec.Body)
	}
}

// restoreRequest builds a multipart restore upload with the settings
// section selected.
func restoreRequest(t *testing.T, file string, htmx bool) *http.Request {
	t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	w.WriteField("sec_settings", "on")
	w.WriteField("sec_models", "on")
	fw, err := w.CreateFormFile("file", "backup.json")
	if err != nil {
		t.Fatal(err)
	}
	fw.Write([]byte(file))
	w.Close()
	req := httptest.NewRequest("POST", "/api/restore", &body)
	req.Header.Set("Content-Type", w.FormDataContentType())
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	return req
}

// A restore of a broken file must say why and change nothing, even when
// part of the file (here the settings section) would have applied. htmx
// does not show non-2xx answers, so the htmx form gets 200 with the error
// in the report; other clients get a 400.
func TestRestoreMalformedUploadChangesNothing(t *testing.T) {
	files := map[string]string{
		"not JSON":        `this is not a backup`,
		"missing fields":  `{"version":1,"settings":{"models_max":7},"model_configs":[{"model_id":""}]}`,
		"unknown version": `{"version":99,"settings":{"models_max":7}}`,
	}
	for name, file := range files {
		t.Run(name, func(t *testing.T) {
			s := newTestServer(t)
			s.cfg.ModelsMax = 2
			addV1Model(t, s, "m", "org/M-GGUF", models.ModelConfig{Enabled: true, ContextSize: 8192})

			rec := serve(s, restoreRequest(t, file, false))
			if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"error"`) {
				t.Errorf("JSON client: HTTP %d: %s", rec.Code, rec.Body)
			}
			rec = serve(s, restoreRequest(t, file, true))
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "&#9888;") {
				t.Errorf("htmx client: HTTP %d, want 200 with the error shown: %s", rec.Code, rec.Body)
			}

			if s.cfg.ModelsMax != 2 {
				t.Errorf("models_max changed to %d", s.cfg.ModelsMax)
			}
			if cfg, _ := s.registry.GetConfig("m"); cfg.ContextSize != 8192 {
				t.Errorf("model config changed: context %d", cfg.ContextSize)
			}
		})
	}
}

// A build request for a profile that does not exist is refused at once
// with the reason, and no build record is created for it.
func TestTriggerBuildUnknownProfile(t *testing.T) {
	s := newTestServer(t)

	req := httptest.NewRequest("POST", "/api/builds", strings.NewReader(`{"profile":"no-such-profile"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := serve(s, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unknown profile") {
		t.Errorf("JSON: HTTP %d: %s", rec.Code, rec.Body)
	}

	req = httptest.NewRequest("POST", "/api/builds", strings.NewReader("profile=no-such-profile"))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec = serve(s, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unknown profile") {
		t.Errorf("form: HTTP %d: %s", rec.Code, rec.Body)
	}

	if builds := s.builder.List(); len(builds) != 0 {
		t.Errorf("a refused build left records: %+v", builds)
	}
}

// Deleting a build id that does not exist is a 404 and leaves the real
// builds alone.
func TestDeleteUnknownBuild(t *testing.T) {
	s := newTestServer(t)
	s.builder = testBuilder(t, testBuild("b1", t.TempDir()))

	rec := serve(s, httptest.NewRequest("DELETE", "/api/builds/nope", nil))
	if rec.Code != http.StatusNotFound {
		t.Errorf("HTTP %d: %s, want 404", rec.Code, rec.Body)
	}
	if _, ok := s.builder.Find("b1"); !ok {
		t.Error("an unrelated build was removed")
	}
}

// While a benchmark job holds the router, every action that would start
// or change it is refused with 409, so a cell never measures a config
// other than the one it reports. The server has a runnable build, so a
// missing guard would really start the router.
func TestRouterActionsRefusedWhileAJobHoldsTheRouter(t *testing.T) {
	s := newTestServer(t)
	s.builder = testBuilder(t, testBuild("b1", fakeLlamaServerDir(t)))
	t.Cleanup(func() { s.process.Stop() })
	addV1Model(t, s, "m", "org/M-GGUF", models.ModelConfig{Enabled: true})
	s.env = &jobEnv{s: s, ownsRouter: true}

	actions := []struct{ method, path string }{
		{"POST", "/api/service/start"},
		{"POST", "/api/service/restart"},
		{"PUT", "/api/models/m/activate"},
		{"DELETE", "/api/models/m/activate"},
		{"POST", "/api/restore"},
	}
	for _, a := range actions {
		rec := serve(s, httptest.NewRequest(a.method, a.path, nil))
		if rec.Code != http.StatusConflict {
			t.Errorf("%s %s: HTTP %d: %s, want 409", a.method, a.path, rec.Code, rec.Body)
		}
	}
	if st := s.process.GetStatus().State; st != process.StateStopped {
		t.Errorf("router state = %s after refused actions, want stopped", st)
	}
}

// Stop is the user's way out of a stuck job holding all the GPU memory,
// so it is deliberately not refused while a job holds the router.
func TestServiceStopAllowedWhileAJobHoldsTheRouter(t *testing.T) {
	s := newTestServer(t)
	t.Cleanup(func() { s.process.Stop() })
	if err := s.process.Start(process.RouterConfig{
		BinaryPath: filepath.Join(fakeLlamaServerDir(t), "llama-server"),
		Host:       "127.0.0.1", Port: 1, ModelsMax: 1,
	}); err != nil {
		t.Fatalf("start: %v", err)
	}
	s.env = &jobEnv{s: s, ownsRouter: true}

	rec := serve(s, httptest.NewRequest("POST", "/api/service/stop", nil))
	if rec.Code != http.StatusOK {
		t.Errorf("stop: HTTP %d: %s, want 200", rec.Code, rec.Body)
	}
	if st := s.process.GetStatus().State; st != process.StateStopped {
		t.Errorf("router state = %s after stop, want stopped", st)
	}
}

// A log stream is an open request; when the browser closes the tab the
// request context ends and the handler must return, or every closed tab
// leaves a goroutine and a log subscriber behind.
func TestStreamLinesStopsWhenTheRequestEnds(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan string)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		StreamLines(rec, ctx, ch, "finished")
		close(done)
	}()

	ch <- "first line" // unbuffered: returns once StreamLines has it
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("StreamLines kept running after the request context ended")
	}

	body := rec.Body.String()
	if !strings.Contains(body, "data: first line\n") {
		t.Errorf("line not sent: %q", body)
	}
	if strings.Contains(body, "event: done") {
		t.Errorf("a cancelled stream must not claim the source finished: %q", body)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q", ct)
	}
}

// When the source closes its channel (the router exited), the client is
// told with a done event carrying the message.
func TestStreamLinesSendsDoneWhenTheSourceCloses(t *testing.T) {
	ch := make(chan string, 1)
	ch <- "last line"
	close(ch)
	rec := httptest.NewRecorder()
	StreamLines(rec, context.Background(), ch, "Router exited")
	if body := rec.Body.String(); !strings.HasSuffix(body, "event: done\ndata: Router exited\n\n") {
		t.Errorf("stream did not end with the done event: %q", body)
	}
}

// The sampling form leaves a field blank to mean "use the model's
// default". Blank and unparsable input must both give nil, never zero,
// which would be sent to llama-server as a real setting.
func TestParseOptionalNumbers(t *testing.T) {
	ints := map[string]*int{
		"":     nil,
		"   ":  nil,
		"40":   ptr(40),
		" 40 ": ptr(40),
		"-1":   ptr(-1),
		"0":    ptr(0),
		"4.5":  nil,
		"abc":  nil,
	}
	for in, want := range ints {
		got := parseOptionalInt(in)
		if (got == nil) != (want == nil) || (got != nil && *got != *want) {
			t.Errorf("parseOptionalInt(%q) = %v, want %v", in, deref(got), deref(want))
		}
	}
	floats := map[string]*float64{
		"":      nil,
		" ":     nil,
		"0.7":   ptr(0.7),
		" 1e-2": ptr(0.01),
		"0":     ptr(0.0),
		"x":     nil,
		"0.7.1": nil,
	}
	for in, want := range floats {
		got := parseOptionalFloat(in)
		if (got == nil) != (want == nil) || (got != nil && *got != *want) {
			t.Errorf("parseOptionalFloat(%q) = %v, want %v", in, deref(got), deref(want))
		}
	}
}

// deref formats an optional value for a test message.
func deref[T any](p *T) any {
	if p == nil {
		return "nil"
	}
	return *p
}
