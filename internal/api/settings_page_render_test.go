package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/web"
)

// Full-page smoke test for the reworked runtime environment section: no
// api test constructs the real server, so without this a parse/execute
// error in settings.html would only surface at runtime.
func TestSettingsPageRendersEnvSection(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse base: %v", err)
	}
	page, err := template.Must(base.Clone()).ParseFS(web.Templates, "templates/settings.html")
	if err != nil {
		t.Fatalf("parse settings.html: %v", err)
	}

	// Mirrors the anonymous struct in handleSettingsPage.
	data := struct {
		pageData
		ProxyEndpoint    string
		LlamaPort        int
		HasAPIKey        bool
		HasHFToken       bool
		HasExtURL        bool
		ExternalURL      string
		DataDir          string
		ModelsDir        string
		DefaultModelsDir string
		AutoStart        bool
		RuntimeEnvOpts   []config.RuntimeEnvOption
		RuntimeEnv       map[string]string
		RuntimeEnvExtra  string
		EnvBackends      []string
		ActiveBackend    string
		EnvWarnings      []string
		EffectiveEnv     []envLine
	}{
		pageData:        pageData{Title: "Settings", Nav: "settings"},
		RuntimeEnvOpts:  config.RuntimeEnvOptions(),
		RuntimeEnv:      map[string]string{"ROCBLAS_USE_HIPBLASLT": "1"},
		RuntimeEnvExtra: "FOO=bar",
		EnvBackends:     config.RuntimeEnvBackends(),
		ActiveBackend:   "rocm",
		EnvWarnings:     []string{"CUDA_DEVICE_ORDER: pinned by the process manager"},
		EffectiveEnv:    []envLine{{Text: "ROCBLAS_USE_HIPBLASLT=1"}},
	}

	var buf bytes.Buffer
	if err := page.ExecuteTemplate(&buf, "layout", data); err != nil {
		t.Fatalf("execute settings page: %v", err)
	}
	out := buf.String()
	for _, want := range []string{
		"runtime_env_extra",         // free-form textarea present
		"FOO=bar",                   // saved extra text round-trips
		"Effective environment",     // preview disclosure present
		"ROCBLAS_USE_HIPBLASLT",     // curated rows rendered
		"GGML_VK_DISABLE_COOPMAT",   // every backend's rows are in the DOM (view filter is client-side)
		"CUDA_DEVICE_ORDER",         // warning surfaced on page load
		`id="effective-env"`,        // OOB swap target exists in the page
		`id="env-backend-select"`,   // backend view selector present
		"rocm (active build)",       // active backend marked in the selector
		`data-backends="cuda rocm"`, // rows carry the filter attribute
		`data-set="1"`,              // set variables marked always-visible
		"filterEnvRows",             // the filter script shipped with the page
	} {
		if !strings.Contains(out, want) {
			t.Errorf("settings page missing %q", want)
		}
	}
}
