package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/web"
)

// The runtime_env_status partial and its effective_env helper are what
// the save handler returns; a parse or execute error there turns every
// env save into a blank status box.
func TestRuntimeEnvStatusPartialRenders(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}

	data := struct {
		Warnings  []string
		Effective []envLine
	}{
		Warnings: []string{"HSA_OVERRIDE_GFX_VERSION: harmful on supported GPUs"},
		Effective: []envLine{
			{Text: "GGML_CUDA_DISABLE_GRAPHS=1"},
			{Text: "FOO=bar", Overridden: "FOO=baz"},
		},
	}

	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "runtime_env_status", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"Settings saved.",
		"HSA_OVERRIDE_GFX_VERSION",
		"GGML_CUDA_DISABLE_GRAPHS=1",
		"FOO=baz", // the inherited value that wins is named
		`id="effective-env"`,
		"hx-swap-oob",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("rendered partial missing %q:\n%s", want, out)
		}
	}
}

// An empty effective environment must render the explanatory placeholder,
// not an empty box.
func TestEffectiveEnvEmptyPlaceholder(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "effective_env", []envLine(nil)); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(buf.String(), "inherits the service environment unchanged") {
		t.Errorf("empty preview should explain itself, got: %s", buf.String())
	}
}
