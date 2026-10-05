package api

import (
	"html/template"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
	"github.com/tmac1973/llama-toolchest/internal/process"
	"github.com/tmac1973/llama-toolchest/web"
)

// testTemplates parses the page templates the way the server does, with
// the server's template functions. With no patterns it parses the layout
// and every partial, which is what most render tests need.
func testTemplates(t *testing.T, patterns ...string) *template.Template {
	t.Helper()
	if len(patterns) == 0 {
		patterns = []string{"templates/layout.html", "templates/partials/*.html"}
	}
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates, patterns...)
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	return base
}

// newTestServer returns a Server with everything a handler test usually
// touches: a config and registry over a temporary data directory, a
// builder with no builds, a monitor that never polls (so no GPUs), a
// stopped process manager, and parsed pages. A test changes the fields it
// cares about afterwards. Keeping one base means a new Server field is
// added here once, not to every test's own constructor.
func newTestServer(t *testing.T) *Server {
	t.Helper()
	dir := t.TempDir()
	s := &Server{
		cfg:         &config.Config{DataDir: dir},
		registry:    models.NewRegistry(dir, filepath.Join(dir, "models")),
		builder:     builder.NewBuilder(dir),
		monitor:     monitor.New(time.Hour),
		process:     process.NewManager(),
		dirtyModels: map[string]bool{},
	}
	s.pages = s.parseTemplates()
	return s
}

// ptr returns a pointer to v, for the optional (pointer) fields of
// overrides and capabilities.
func ptr[T any](v T) *T { return &v }
