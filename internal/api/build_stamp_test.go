package api

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// notRecorded is the exact sentence an unstamped build must carry, in both the
// table and the info modal. Asserted in full rather than as a fragment, so
// rewording one of the two call sites fails the test instead of passing on a
// partial match.
const notRecorded = "Not recorded — this build predates build-environment tracking."

// stampServer builds a Server whose builder holds the given builds and whose
// idea of the current toolchain is fixed by the test. Fixing it is the point:
// without the seam, "matching" and "differing" would be decided by whichever
// ROCm happens to be installed on the machine running `go test`.
func stampServer(t *testing.T, current string, builds []builder.BuildResult) *Server {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(builds)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Server{
		registry:        models.NewRegistry(t.TempDir(), t.TempDir()),
		builder:         builder.NewBuilder(dir),
		currentBuildEnv: func(string) string { return current },
	}
	s.pages = s.parseTemplates()
	return s
}

// The three builds are: one stamped and matching, one stamped and differing,
// one with no stamp at all. Exactly one row may be flagged — the differing one.
// The unstamped row must NOT be flagged: Phase 04's rule is that an unknown
// stamp is never reported as a mismatch, because such a build may work fine.
func TestBuildsListFlagsOnlyAGenuineMismatch(t *testing.T) {
	s := stampServer(t, "rocm 10.0.0", []builder.BuildResult{
		{ID: "b-match", Profile: "rocm", GitSHA: "aaaaaaaaaa", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 10.0.0"},
		{ID: "b-differs", Profile: "rocm", GitSHA: "bbbbbbbbbb", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
		{ID: "b-unstamped", Profile: "rocm", GitSHA: "cccccccccc", Status: builder.BuildStatusSuccess},
	})

	req := httptest.NewRequest("GET", "/api/builds", nil)
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleListBuilds(rec, req)
	out := rec.Body.String()

	if !strings.Contains(out, "<th title=") || !strings.Contains(out, ">Built against</th>") {
		t.Errorf("the Built against column header is missing\n%s", out)
	}
	// One warning mark, on one row.
	if n := strings.Count(out, "&#9888;"); n != 1 {
		t.Errorf("%d rows carry the warning mark, want exactly 1\n%s", n, out)
	}

	rows := map[string]string{}
	for _, chunk := range strings.Split(out, "<tr>")[1:] {
		for _, id := range []string{"b-match", "b-differs", "b-unstamped"} {
			if strings.Contains(chunk, ">"+id+"<") {
				rows[id] = chunk
			}
		}
	}
	if len(rows) != 3 {
		t.Fatalf("found %d of 3 build rows\n%s", len(rows), out)
	}

	if strings.Contains(rows["b-match"], "&#9888;") {
		t.Error("a build matching the running toolchain must not be flagged")
	}
	if !strings.Contains(rows["b-match"], "rocm 10.0.0") {
		t.Errorf("the matching row does not show its stamp\n%s", rows["b-match"])
	}

	if !strings.Contains(rows["b-differs"], "&#9888;") {
		t.Errorf("the differing build is not flagged\n%s", rows["b-differs"])
	}
	// The tooltip must name both sides, so the reader knows what to do.
	if !strings.Contains(rows["b-differs"], "rocm 7.2.4") || !strings.Contains(rows["b-differs"], "rocm 10.0.0") {
		t.Errorf("the flagged row's tooltip does not name both versions\n%s", rows["b-differs"])
	}
	// "may fail to load", not "will not run": a cross-version build was
	// measured loading successfully on this project's own images, so the
	// wording must not promise a failure it cannot guarantee.
	if !strings.Contains(rows["b-differs"], "may fail to load") ||
		!strings.Contains(rows["b-differs"], "rebuild it if the server does not start") {
		t.Errorf("the flagged row does not describe the risk accurately\n%s", rows["b-differs"])
	}

	if strings.Contains(rows["b-unstamped"], "&#9888;") {
		t.Error("an unstamped build must not be flagged — unknown is not a mismatch")
	}
	if !strings.Contains(rows["b-unstamped"], notRecorded) {
		t.Errorf("the unstamped row does not carry the not-recorded tooltip verbatim\n%s", rows["b-unstamped"])
	}
	if !strings.Contains(rows["b-unstamped"], "—") {
		t.Errorf("the unstamped row does not render an em-dash\n%s", rows["b-unstamped"])
	}
}

// The info modal reads the same two fields from buildRowFor as the table does.
// This is what proves the second call site is wired to the shared constant
// rather than carrying its own copy of the wording.
func TestBuildInfoShowsTheSameStampAsTheTable(t *testing.T) {
	s := stampServer(t, "rocm 10.0.0", []builder.BuildResult{
		{ID: "b-differs", Profile: "rocm", GitSHA: "bbbbbbbbbb", Status: builder.BuildStatusSuccess, BuiltAgainst: "rocm 7.2.4"},
		{ID: "b-unstamped", Profile: "rocm", GitSHA: "cccccccccc", Status: builder.BuildStatusSuccess},
	})

	// The handler is called directly with a chi route context rather than
	// through buildRouter(), which constructs a proxy handler and needs a
	// fully wired Server this test has no use for.
	info := func(id string) string {
		t.Helper()
		req := httptest.NewRequest("GET", "/api/builds/"+id+"/info", nil)
		req.Header.Set("HX-Request", "true")
		rctx := chi.NewRouteContext()
		rctx.URLParams.Add("id", id)
		req = req.WithContext(context.WithValue(req.Context(), chi.RouteCtxKey, rctx))
		rec := httptest.NewRecorder()
		s.handleBuildInfo(rec, req)
		if rec.Code != 200 {
			t.Fatalf("info for %s: HTTP %d — %s", id, rec.Code, rec.Body.String())
		}
		return rec.Body.String()
	}

	out := info("b-differs")
	if !strings.Contains(out, "Built against") {
		t.Errorf("the info modal has no Built against row\n%s", out)
	}
	if !strings.Contains(out, "rocm 7.2.4") || !strings.Contains(out, "&#9888;") {
		t.Errorf("the info modal does not show the stamp and the flag\n%s", out)
	}

	out = info("b-unstamped")
	if !strings.Contains(out, notRecorded) {
		t.Errorf("the info modal's unstamped wording differs from the table's\n%s", out)
	}
	if strings.Contains(out, "&#9888;") {
		t.Error("an unstamped build must not be flagged in the info modal either")
	}
}

// A build record written before this field existed has no built_against key.
// It must load without complaint and read as unknown — not as a mismatch, and
// not as an error.
func TestBuildsWithoutAStampStillLoad(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	legacy := `[{"id":"b10453-rocm-optimized","profile":"rocm","git_sha":"deadbeefcafe","status":"success"}]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}
	b := builder.NewBuilder(dir)
	list := b.List()
	if len(list) != 1 {
		t.Fatalf("loaded %d builds from a pre-stamp builds.json, want 1", len(list))
	}
	if list[0].BuiltAgainst != "" {
		t.Errorf("BuiltAgainst = %q on a legacy record, want empty", list[0].BuiltAgainst)
	}
	if builder.StampMismatch(list[0].BuiltAgainst, "rocm 10.0.0") {
		t.Error("a legacy record was reported as a mismatch")
	}
}
