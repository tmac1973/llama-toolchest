package api

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// sthLlamaPort starts a fake llama-server on 127.0.0.1 and returns its
// port and the paths it was asked for.
func sthLlamaPort(t *testing.T, status int) (int, *atomic.Value) {
	t.Helper()
	var path atomic.Value
	path.Store("")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path.Store(r.URL.Path)
		w.WriteHeader(status)
	}))
	t.Cleanup(srv.Close)
	_, port, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	n, _ := strconv.Atoi(port)
	return n, &path
}

// sthClosedPort returns a port on 127.0.0.1 that nothing listens on.
func sthClosedPort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	return port
}

func sthTestConnection(s *Server, htmx bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/api/settings/test-connection", nil)
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	s.handleTestConnection(rec, req)
	return rec
}

// The connection test asks llama-server for /v1/models on the configured
// port and reports success only for a 200.
func TestTestConnectionReportsSuccess(t *testing.T) {
	port, path := sthLlamaPort(t, http.StatusOK)
	s := newSettingsServer(t, &config.Config{LlamaPort: port})

	rec := sthTestConnection(s, false)
	var res connectionTestResult
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if !res.OK || res.Error != "" || res.Latency == "" {
		t.Errorf("result = %+v, want OK with a latency", res)
	}
	if p := path.Load().(string); p != "/v1/models" {
		t.Errorf("asked for %q, want /v1/models", p)
	}

	rec = sthTestConnection(s, true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Connected") {
		t.Errorf("htmx: status %d body %q, want 200 Connected", rec.Code, rec.Body.String())
	}
}

// A server that answers with an error status, and a port with nothing
// listening, are both failures. The htmx answer is still 200, with the
// reason in the page, because htmx would not swap a non-2xx answer into
// the result area.
func TestTestConnectionReportsFailures(t *testing.T) {
	badPort, _ := sthLlamaPort(t, http.StatusServiceUnavailable)
	for _, tc := range []struct {
		name string
		port int
		want string
	}{
		{"error status", badPort, "HTTP 503"},
		{"nothing listening", sthClosedPort(t), "refused"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newSettingsServer(t, &config.Config{LlamaPort: tc.port})
			var res connectionTestResult
			if err := json.Unmarshal(sthTestConnection(s, false).Body.Bytes(), &res); err != nil {
				t.Fatal(err)
			}
			if res.OK || !strings.Contains(res.Error, tc.want) {
				t.Errorf("result = %+v, want a failure mentioning %q", res, tc.want)
			}
			rec := sthTestConnection(s, true)
			if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Failed") || !strings.Contains(rec.Body.String(), tc.want) {
				t.Errorf("htmx: status %d body %q, want 200 Failed with the reason", rec.Code, rec.Body.String())
			}
		})
	}
}

// validateModelsDir accepts an empty value (the default folder) and an
// existing absolute directory, and refuses everything else with a reason
// the settings page can show.
func TestValidateModelsDir(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "a-file")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, path, wantErr string
	}{
		{"empty means the default", "", ""},
		{"existing directory", dir, ""},
		{"relative path", "models", "must be an absolute path"},
		{"missing directory", filepath.Join(dir, "missing"), "path does not exist"},
		{"a file", file, "not a directory"},
	} {
		err := validateModelsDir(tc.path)
		switch {
		case tc.wantErr == "" && err != nil:
			t.Errorf("%s: unexpected error %v", tc.name, err)
		case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
			t.Errorf("%s: error %v, want %q", tc.name, err, tc.wantErr)
		}
	}
}

// A directory that cannot be read is refused. A directory that can be
// read but not written is accepted: the check only covers reading, so a
// read-only collection of models can still be served. (Downloads into
// such a folder fail later, when the download starts.)
func TestValidateModelsDirPermissions(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permission checks do not apply to root")
	}
	unreadable := filepath.Join(t.TempDir(), "unreadable")
	readOnly := filepath.Join(t.TempDir(), "read-only")
	for _, d := range []string{unreadable, readOnly} {
		if err := os.Mkdir(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Chmod(unreadable, 0o000); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(readOnly, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		os.Chmod(unreadable, 0o755)
		os.Chmod(readOnly, 0o755)
	})

	if err := validateModelsDir(unreadable); err == nil || !strings.Contains(err.Error(), "not readable") {
		t.Errorf("unreadable directory: error %v, want not readable", err)
	}
	if err := validateModelsDir(readOnly); err != nil {
		t.Errorf("read-only directory: error %v, want it accepted", err)
	}
}

func sthPutForm(s *Server, form url.Values) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	return rec
}

// A models folder that fails the check is refused with 400 and the
// reason, and the saved folder is not changed. Posting the folder that is
// already saved is not checked again, so a folder on a disk that is
// unmounted for a moment does not block saving other settings.
func TestUpdateSettingsChecksTheModelsDir(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	missing := filepath.Join(t.TempDir(), "missing")

	rec := sthPutForm(s, url.Values{"models_dir": {missing}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "models_dir: path does not exist") {
		t.Errorf("status %d body %q, want 400 with the reason", rec.Code, rec.Body.String())
	}
	if s.cfg.ModelsDir != "" {
		t.Errorf("ModelsDir = %q after a refused save", s.cfg.ModelsDir)
	}

	good := t.TempDir()
	if rec := sthPutForm(s, url.Values{"models_dir": {"  " + good + "  "}}); rec.Code != http.StatusOK {
		t.Fatalf("valid folder: status %d body %q", rec.Code, rec.Body.String())
	}
	if s.cfg.ModelsDir != good {
		t.Errorf("ModelsDir = %q, want the trimmed %q", s.cfg.ModelsDir, good)
	}

	s.cfg.ModelsDir = missing // saved earlier, gone now
	if rec := sthPutForm(s, url.Values{"models_dir": {missing}, "hf_token": {"hf_new"}}); rec.Code != http.StatusOK {
		t.Errorf("unchanged folder: status %d body %q, want the save to go through", rec.Code, rec.Body.String())
	}
	if s.cfg.HFToken != "hf_new" {
		t.Errorf("HFToken = %q, want the other field saved", s.cfg.HFToken)
	}
}

// applyGenPorts refuses ranges that are out of bounds or that include a
// port toolchest already uses, and leaves the saved range unchanged when
// it refuses.
func TestApplyGenPortsValidates(t *testing.T) {
	for _, tc := range []struct {
		name         string
		start, count *int
		wantErr      string
	}{
		{"start below 1024", ptr(80), nil, "must be between 1024 and 65535"},
		{"start above 65535", ptr(70000), nil, "must be between 1024 and 65535"},
		{"count zero", nil, ptr(0), "port count 0 must be between 1 and 32"},
		{"count too large", nil, ptr(33), "port count 33 must be between 1 and 32"},
		{"range past 65535", ptr(65530), ptr(8), "goes past 65535"},
		{"includes the llama-server port", ptr(9000), ptr(8), "includes the llama-server port 9005"},
		{"includes the web UI port", ptr(2995), ptr(8), "includes the web UI port 3000"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newSettingsServer(t, &config.Config{LlamaPort: 9005, ListenAddr: "0.0.0.0:3000", GenPortStart: 8200, GenPortCount: 4})
			err := s.applyGenPorts(tc.start, tc.count)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %v, want %q", err, tc.wantErr)
			}
			if s.cfg.GenPortStart != 8200 || s.cfg.GenPortCount != 4 {
				t.Errorf("range changed to %d+%d by a refused request", s.cfg.GenPortStart, s.cfg.GenPortCount)
			}
		})
	}
}

// A nil value keeps the current one, and with nothing saved the current
// range is the default 8081, 8 ports. Changing only the count therefore
// keeps the default start.
func TestApplyGenPortsKeepsTheOtherValue(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, ListenAddr: ":3000"})
	if err := s.applyGenPorts(nil, ptr(4)); err != nil {
		t.Fatal(err)
	}
	if s.cfg.GenPortStart != config.DefaultGenPortStart || s.cfg.GenPortCount != 4 {
		t.Errorf("range = %d+%d, want %d+4", s.cfg.GenPortStart, s.cfg.GenPortCount, config.DefaultGenPortStart)
	}
	if err := s.applyGenPorts(ptr(9100), nil); err != nil {
		t.Fatal(err)
	}
	if s.cfg.GenPortStart != 9100 || s.cfg.GenPortCount != 4 {
		t.Errorf("range = %d+%d, want 9100+4", s.cfg.GenPortStart, s.cfg.GenPortCount)
	}
}

// Inside a container the published ports fix the range, so every change
// is refused, even a valid one.
func TestApplyGenPortsRefusedInAContainer(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080})
	s.inContainerFn = func() bool { return true }
	err := s.applyGenPorts(ptr(9100), ptr(4))
	if err == nil || !strings.Contains(err.Error(), "inside a container") {
		t.Errorf("error %v, want the container refusal", err)
	}
	if s.cfg.GenPortStart != 0 || s.cfg.GenPortCount != 0 {
		t.Errorf("range changed to %d+%d inside a container", s.cfg.GenPortStart, s.cfg.GenPortCount)
	}
}

// formGenPorts reads the two form fields, trims spaces, leaves an absent
// field as nil (keep the current value), and names the field that is not
// a number.
func TestFormGenPorts(t *testing.T) {
	parse := func(form url.Values) (*int, *int, error) {
		req := httptest.NewRequest(http.MethodPut, "/", strings.NewReader(form.Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		if err := req.ParseForm(); err != nil {
			t.Fatal(err)
		}
		return formGenPorts(req)
	}

	start, count, err := parse(url.Values{"gen_port_start": {" 9100 "}})
	if err != nil || start == nil || *start != 9100 || count != nil {
		t.Errorf("start only: %v %v %v, want 9100 and no count", start, count, err)
	}
	start, count, err = parse(url.Values{"gen_port_start": {"9100"}, "gen_port_count": {"6"}})
	if err != nil || *start != 9100 || count == nil || *count != 6 {
		t.Errorf("both: %v %v %v", start, count, err)
	}
	for _, field := range []string{"gen_port_start", "gen_port_count"} {
		if _, _, err := parse(url.Values{field: {"eight"}}); err == nil || err.Error() != field+": not a number" {
			t.Errorf("%s: error %v, want %q", field, err, field+": not a number")
		}
	}
}

// A refused port range through the settings form or JSON is a 400 with
// the reason, and the other fields of the same request are not saved
// either, so a save is all or nothing.
func TestUpdateSettingsRefusedPortRangeSavesNothing(t *testing.T) {
	s := newSettingsServer(t, &config.Config{LlamaPort: 8080, HFToken: "hf_old"})

	rec := sthPutForm(s, url.Values{"gen_port_start": {"abc"}, "hf_token": {"hf_form"}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "gen_port_start: not a number") {
		t.Errorf("form, not a number: status %d body %q", rec.Code, rec.Body.String())
	}
	rec = sthPutForm(s, url.Values{"gen_port_start": {"8077"}, "gen_port_count": {"8"}, "hf_token": {"hf_form"}})
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "llama-server port 8080") {
		t.Errorf("form, conflict: status %d body %q", rec.Code, rec.Body.String())
	}

	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(`{"gen_port_count":99,"hf_token":"hf_json"}`))
	req.Header.Set("Content-Type", "application/json")
	rec = httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "port count 99") {
		t.Errorf("JSON: status %d body %q", rec.Code, rec.Body.String())
	}

	if s.cfg.HFToken != "hf_old" || s.cfg.GenPortStart != 0 || s.cfg.GenPortCount != 0 {
		t.Errorf("after refused saves: token %q range %d+%d, want nothing changed", s.cfg.HFToken, s.cfg.GenPortStart, s.cfg.GenPortCount)
	}
	if _, err := os.Stat(s.configPath); !os.IsNotExist(err) {
		t.Errorf("config file written by a refused save (err %v)", err)
	}
}

// The proxy endpoint is the external URL with /v1 added, without a
// double slash when the URL ends in one or more slashes. This is the
// address users copy into their clients, so a "//v1" would break them.
func TestProxyEndpoint(t *testing.T) {
	for _, tc := range []struct{ external, want string }{
		{"http://localhost:3000", "http://localhost:3000/v1"},
		{"http://localhost:3000/", "http://localhost:3000/v1"},
		{"https://ai.example.com/tools//", "https://ai.example.com/tools/v1"},
	} {
		s := &Server{cfg: &config.Config{ExternalURL: tc.external}}
		if got := s.proxyEndpoint(); got != tc.want {
			t.Errorf("external %q: %q, want %q", tc.external, got, tc.want)
		}
	}

	// The settings API reports the same value.
	s := newSettingsServer(t, &config.Config{ExternalURL: "http://box:3000/"})
	rec := httptest.NewRecorder()
	s.handleGetSettings(rec, httptest.NewRequest(http.MethodGet, "/api/settings", nil))
	var resp settingsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.ProxyEndpoint != "http://box:3000/v1" {
		t.Errorf("proxy_endpoint = %q", resp.ProxyEndpoint)
	}
}
