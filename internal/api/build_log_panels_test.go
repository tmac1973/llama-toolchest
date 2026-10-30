package api

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// Each running build gets its own log panel with its ID and its own
// stream, so two builds in parallel each show their log.
func TestBuildLogPanelsOnePerBuild(t *testing.T) {
	s := newTestServer(t)
	rec := httptest.NewRecorder()
	s.renderPartial(rec, "build_logs", []builder.BuildResult{
		{ID: "ace-master-abc1234-rocm", Status: builder.BuildStatusBuilding},
		{ID: "b11960-rocm", Status: builder.BuildStatusBuilding},
	})
	out := rec.Body.String()
	for _, id := range []string{"ace-master-abc1234-rocm", "b11960-rocm"} {
		if !strings.Contains(out, "Build Log — "+id) || !strings.Contains(out, `sse-connect="/api/builds/`+id+`/logs"`) {
			t.Errorf("no panel for %s:\n%s", id, out)
		}
	}
	if n := strings.Count(out, `class="log-panel"`); n != 2 {
		t.Errorf("%d panels, want 2", n)
	}
	if strings.Contains(out, `id="log-output"`) {
		t.Error("panels share an element id")
	}

	// With no build, the log area stays empty.
	rec = httptest.NewRecorder()
	s.handleActiveBuildLog(rec, httptest.NewRequest("GET", "/api/builds/active-log", nil))
	if strings.TrimSpace(rec.Body.String()) != "" {
		t.Errorf("log area without builds: %q", rec.Body.String())
	}
}

// The start stream refuses a malformed token.
func TestBuildStartLogsRefusesBadToken(t *testing.T) {
	s := newTestServer(t)
	r := s.buildRouter()
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest("GET", "/api/builds/start/NOT-A-TOKEN/logs", nil))
	if rec.Code != 404 {
		t.Errorf("status %d, want 404", rec.Code)
	}
}
