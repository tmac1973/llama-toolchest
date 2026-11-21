package builder

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

var testExtras = []ExtraFiles{{Src: filepath.Join("examples", "server", "recipes"), Dst: "recipes", Glob: "*.json"}}

func writeFiles(t *testing.T, dir string, names ...string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range names {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func listDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

// A build keeps the checkout's recipe files, and only those.
func TestCopyExtras(t *testing.T) {
	src, out := t.TempDir(), t.TempDir()
	recipes := filepath.Join(src, "examples", "server", "recipes")
	writeFiles(t, recipes, "a.json", "b.json", "README.md")
	var log []string
	if err := copyExtras(src, out, testExtras, func(s string) { log = append(log, s) }); err != nil {
		t.Fatal(err)
	}
	if got := listDir(t, filepath.Join(out, "recipes")); !slices.Equal(got, []string{"a.json", "b.json"}) {
		t.Errorf("recipes = %v", got)
	}
	if !slices.ContainsFunc(log, func(s string) bool { return strings.Contains(s, "Copied 2 files to recipes") }) {
		t.Errorf("log = %v", log)
	}

	// A forced rebuild after upstream deleted b.json leaves only a.json.
	if err := os.Remove(filepath.Join(recipes, "b.json")); err != nil {
		t.Fatal(err)
	}
	if err := copyExtras(src, out, testExtras, func(string) {}); err != nil {
		t.Fatal(err)
	}
	if got := listDir(t, filepath.Join(out, "recipes")); !slices.Equal(got, []string{"a.json"}) {
		t.Errorf("recipes after rebuild = %v", got)
	}
}

// An old checkout without recipe files builds fine and says so.
func TestCopyExtrasMissing(t *testing.T) {
	src, out := t.TempDir(), t.TempDir()
	var log []string
	if err := copyExtras(src, out, testExtras, func(s string) { log = append(log, s) }); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(out, "recipes")); !os.IsNotExist(err) {
		t.Errorf("recipes folder made for a checkout without recipes: %v", err)
	}
	if len(log) != 1 || !strings.Contains(log[0], "No examples/server/recipes") {
		t.Errorf("log = %v", log)
	}
}

// Only the sd project keeps extra files.
func TestProjectExtras(t *testing.T) {
	for _, p := range projects() {
		if (len(p.Extras) > 0) != (p.ID == ProjectSDCpp) {
			t.Errorf("%s has extras %v", p.ID, p.Extras)
		}
	}
}

// OnChange runs after a build is deleted.
func TestOnChangeAfterDelete(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "config"), 0o755); err != nil {
		t.Fatal(err)
	}
	records := `[{"id":"sd-x-rocm","project":"stable-diffusion.cpp","profile":"rocm","status":"success"}]`
	if err := os.WriteFile(filepath.Join(dir, "config", "builds.json"), []byte(records), 0o644); err != nil {
		t.Fatal(err)
	}
	b := NewBuilder(dir)
	called := make(chan struct{}, 2)
	b.OnChange = func() { called <- struct{}{} }
	if err := b.Delete("sd-x-rocm"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-called:
	case <-time.After(5 * time.Second):
		t.Fatal("OnChange not called after Delete")
	}
	// finishBuild calls it too.
	b.finishBuild(&BuildResult{ID: "y"}, BuildStatusFailed, "test")
	select {
	case <-called:
	case <-time.After(5 * time.Second):
		t.Fatal("OnChange not called after finishBuild")
	}
}
