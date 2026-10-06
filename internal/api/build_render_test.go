package api

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
)

// The Info dialog shows the cmake flags in a <pre>, sorted, one per line
// with a shell line continuation, so the whitespace in it is what the
// reader sees. A build from before flags were recorded says so instead.
func TestBuildInfoFlagList(t *testing.T) {
	s := newTestServer(t)
	s.builder = testBuilder(t,
		builder.BuildResult{ID: "b1", GitSHA: "abc1234567", CMakeFlags: map[string]string{"B": "ON", "A": "<x>"}},
		builder.BuildResult{ID: "b2", GitSHA: "abc1234567"},
	)
	info := func(id string) string {
		t.Helper()
		req := httptest.NewRequest("GET", "/api/builds/"+id+"/info", nil)
		req.Header.Set("HX-Request", "true")
		rctx := chi.NewRouteContext()
		rctx.URLParams.Add("id", id)
		req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))
		rec := httptest.NewRecorder()
		s.handleBuildInfo(rec, req)
		return rec.Body.String()
	}

	out := info("b1")
	want := `word-break:break-all;">-DA=&lt;x&gt; \` + "\n" + `  -DB=ON</pre>`
	if !strings.Contains(out, want) {
		t.Errorf("flag list not sorted one per line and escaped once:\n%s", out)
	}
	if strings.Contains(out, "not recorded") {
		t.Error("a build with flags says they were not recorded")
	}

	out = info("b2")
	if !strings.Contains(out, "cmake flags not recorded") || strings.Contains(out, "<pre") {
		t.Errorf("a build without flags should say so:\n%s", out)
	}
}
