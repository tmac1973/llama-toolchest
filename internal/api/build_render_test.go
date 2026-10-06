package api

import (
	"context"
	"errors"
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

// The git ref picker puts v* release tags and b* nightlies in separate
// groups, labels a release with the nightly it was cut from, and says
// why a refresh failed when there is nothing cached to show instead.
func TestGitRefOptions(t *testing.T) {
	s := newTestServer(t)
	render := func(refs []string, anchors map[string]int, err error) string {
		t.Helper()
		rec := httptest.NewRecorder()
		s.renderPartial(rec, "git_ref_options", gitRefOptionsFor(refs, anchors, err))
		return strings.Join(strings.Fields(rec.Body.String()), " ")
	}

	out := render([]string{"v0.2.0", "b100"}, map[string]int{"v0.2.0": 10500}, nil)
	want := `<option value="latest">latest</option>` +
		` <optgroup label="Releases"> <option value="v0.2.0">v0.2.0 (b10500)</option> </optgroup>` +
		` <optgroup label="Nightly builds"> <option value="b100">b100</option> </optgroup>`
	if out != want {
		t.Errorf("got\n%s\nwant\n%s", out, want)
	}

	out = render(nil, nil, errors.New("fetch <failed>"))
	if !strings.Contains(out, `<option disabled>— fetch &lt;failed&gt; —</option>`) || strings.Contains(out, "optgroup") {
		t.Errorf("failed refresh with nothing cached:\n%s", out)
	}
	if out := render([]string{"b1"}, nil, errors.New("x")); strings.Contains(out, "disabled") {
		t.Errorf("a failed refresh with cached tags shows the cached tags only:\n%s", out)
	}
}
