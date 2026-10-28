package api

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/config"
)

// The runtime_env_status partial and its effective_env helper are what
// the save handler returns; a parse or execute error there turns every
// env save into a blank status box.
func TestRuntimeEnvStatusPartialRenders(t *testing.T) {
	base := testTemplates(t)

	data := struct {
		Warnings  []string
		Effective []envLine
	}{
		Warnings: []string{"HSA_OVERRIDE_GFX_VERSION: harmful on supported GPUs"},
		Effective: []envLine{
			{Text: "GGML_CUDA_DISABLE_GRAPHS=1"},
			{Text: "FOO=bar", Overridden: "FOO"},
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
		"service environment sets FOO,", // the inherited name is shown, not its value
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
	base := testTemplates(t)
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "effective_env", []envLine(nil)); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(buf.String(), "inherits the service environment unchanged") {
		t.Errorf("empty preview should explain itself, got: %s", buf.String())
	}
}

// The inherited value can be a token from the container or systemd
// environment; the settings page shows only that the name is set.
func TestEffectiveEnvHidesTheInheritedValue(t *testing.T) {
	t.Setenv("LT_TEST_SECRET", "hf_inherited_secret")
	s := &Server{cfg: &config.Config{RuntimeEnvExtra: "LT_TEST_SECRET=configured"}}
	lines := effectiveEnvLines(s.cfg.EnvSet())
	if len(lines) != 1 || lines[0].Overridden != "LT_TEST_SECRET" {
		t.Fatalf("lines = %+v, want LT_TEST_SECRET marked as overridden", lines)
	}
	for _, l := range lines {
		if strings.Contains(l.Text+l.Overridden, "hf_inherited_secret") {
			t.Errorf("inherited value shown: %+v", l)
		}
	}
}

// A secret-looking value typed into the extra lines is masked in the
// effective-environment preview.
func TestEffectiveEnvMasksSecretValues(t *testing.T) {
	s := &Server{cfg: &config.Config{RuntimeEnvExtra: "LT_TEST_HF_TOKEN=hf_typed_secret\nLT_PLAIN=1"}}
	var texts []string
	for _, l := range effectiveEnvLines(s.cfg.EnvSet()) {
		texts = append(texts, l.Text)
	}
	got := strings.Join(texts, "\n")
	if strings.Contains(got, "hf_typed_secret") || !strings.Contains(got, "LT_TEST_HF_TOKEN="+config.EnvSecretMask) || !strings.Contains(got, "LT_PLAIN=1") {
		t.Errorf("effective env = %q", got)
	}
}
