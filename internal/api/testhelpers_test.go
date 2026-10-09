package api

import (
	"html/template"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
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
		// No ROCm installs unless a test says otherwise, so results do not
		// depend on what the machine running the tests has.
		rocmInstallsFn: func() []builder.ROCmInstall { return nil },
	}
	s.pages = s.parseTemplates()
	return s
}

// ptr returns a pointer to v, for the optional (pointer) fields of
// overrides and capabilities.
func ptr[T any](v T) *T { return &v }

// templateScripts returns the <script> blocks of a page template with its
// Go template actions replaced by 0, so node can parse them. That
// replacement can turn an action inside a string into invalid JS; the
// node test then fails to load, which is loud rather than silent.
func templateScripts(t *testing.T, page string) string {
	t.Helper()
	src := mustRead(t, filepath.Join("..", "..", "web", "templates", page))
	blocks := scriptBlock.FindAllStringSubmatch(src, -1)
	if len(blocks) == 0 {
		t.Fatalf("no <script> block found in %s", page)
	}
	var js strings.Builder
	for _, b := range blocks {
		js.WriteString(templateAction.ReplaceAllString(b[1], "0"))
		js.WriteString("\n")
	}
	return js.String()
}

var (
	scriptBlock    = regexp.MustCompile(`(?s)<script>(.*?)</script>`)
	templateAction = regexp.MustCompile(`\{\{[^}]*\}\}`)
)

// runNodeTest writes files into a temporary directory and runs its run.js
// with node, failing unless the script prints "ALL PASS". It skips the
// test when node is not installed; `make js-test` requires it.
func runNodeTest(t *testing.T, files map[string]string) {
	t.Helper()
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node not installed; skipping JS test")
	}
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	cmd := exec.Command(node, filepath.Join(dir, "run.js"))
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "ALL PASS") {
		t.Fatalf("JS test failed:\n%s", out)
	}
}

// jsTestFile reads a file from web/jstest.
func jsTestFile(t *testing.T, name string) string {
	t.Helper()
	return mustRead(t, filepath.Join("..", "..", "web", "jstest", name))
}
