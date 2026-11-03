package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestNewerRelease(t *testing.T) {
	cases := []struct {
		current, release string
		want             bool
	}{
		{"2.49.0", "v2.50.0", true},
		{"2.49.0", "v2.49.1", true},
		{"2.49.0", "v3.0.0", true},
		{"2.49.0", "v2.49.0", false},
		{"2.50.0", "v2.49.9", false},
		{"2.9.0", "v2.10.0", true},
		{"dev", "v2.50.0", false},
		{"dev-abc1234", "v2.50.0", false},
		{"2.49.0-snapshot+abc1234", "v2.50.0", false},
		{"2.49.0", "", false},
		{"2.49.0", "nightly", false},
	}
	for _, c := range cases {
		if got := newerRelease(c.current, c.release); got != c.want {
			t.Errorf("newerRelease(%q, %q) = %v, want %v", c.current, c.release, got, c.want)
		}
	}
}

// fakeGitHub serves one release and counts how often it is asked.
func fakeGitHub(t *testing.T, status int, body string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func releaseCheckRequest(s *Server) *httptest.ResponseRecorder {
	req := httptest.NewRequest("GET", "/api/release-check", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleReleaseCheck(rec, req)
	return rec
}

func TestReleaseCheckBadge(t *testing.T) {
	gh, calls := fakeGitHub(t, http.StatusOK,
		`{"tag_name":"v2.50.0","html_url":"https://github.com/tmac1973/llama-toolchest/releases/tag/v2.50.0"}`)
	s := newTestServer(t)
	s.version = "2.49.0"
	s.releaseCheck.apiURL = gh.URL

	body := releaseCheckRequest(s).Body.String()
	if !strings.Contains(body, "New version v2.50.0 available") {
		t.Errorf("badge text missing: %q", body)
	}
	if !strings.Contains(body, `href="https://github.com/tmac1973/llama-toolchest/releases/tag/v2.50.0"`) {
		t.Errorf("badge does not link the release page: %q", body)
	}

	// A second page load uses the kept answer.
	releaseCheckRequest(s)
	if n := calls.Load(); n != 1 {
		t.Errorf("GitHub asked %d times, want 1", n)
	}
}

func TestReleaseCheckNoBadge(t *testing.T) {
	release := `{"tag_name":"v2.49.0","html_url":"https://github.com/tmac1973/llama-toolchest/releases/tag/v2.49.0"}`
	cases := []struct {
		name, version string
		status        int
		wantCalls     int32
	}{
		{"up to date", "2.49.0", http.StatusOK, 1},
		{"github error", "2.48.0", http.StatusForbidden, 1},
		{"dev build", "dev-abc1234", http.StatusOK, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			gh, calls := fakeGitHub(t, c.status, release)
			s := newTestServer(t)
			s.version = c.version
			s.releaseCheck.apiURL = gh.URL

			if body := releaseCheckRequest(s).Body.String(); body != "" {
				t.Errorf("want no badge, got %q", body)
			}
			// A failed check is not retried on the next page load either.
			releaseCheckRequest(s)
			if n := calls.Load(); n != c.wantCalls {
				t.Errorf("GitHub asked %d times, want %d", n, c.wantCalls)
			}
		})
	}
}

func TestReleaseCheckUntrustedLink(t *testing.T) {
	gh, _ := fakeGitHub(t, http.StatusOK, `{"tag_name":"v9.0.0","html_url":"javascript:alert(1)"}`)
	s := newTestServer(t)
	s.version = "2.49.0"
	s.releaseCheck.apiURL = gh.URL

	body := releaseCheckRequest(s).Body.String()
	if !strings.Contains(body, `href="`+latestReleasePage+`"`) {
		t.Errorf("want the releases page link, got %q", body)
	}
}
