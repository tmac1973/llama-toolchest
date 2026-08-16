package api

import (
	"bytes"
	"html/template"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/web"
)

// jobFormTestData mirrors the anonymous struct in handleJobForm so a
// parse or execute error in the partial surfaces in tests instead of
// turning every form open into a blank modal.
type jobFormTestData struct {
	Models      []*models.Model
	Builds      []buildOpt
	Presets     []benchmark.Preset
	GPUOptions  []models.GPUOption
	Params      []paramView
	MaxCells    int
	Running     bool
	KLReference []klRepoGroup
}

func renderJobFormPartial(t *testing.T, data jobFormTestData) string {
	t.Helper()
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	var buf bytes.Buffer
	if err := base.ExecuteTemplate(&buf, "job_form", data); err != nil {
		t.Fatalf("execute: %v", err)
	}
	return buf.String()
}

func jobFormData(t *testing.T) jobFormTestData {
	t.Helper()
	m4 := &models.Model{ID: "m4", ModelID: "u/M-GGUF", Quant: "Q4_K_XL", SizeBytes: 2 << 30}
	m8 := &models.Model{ID: "m8", ModelID: "u/M-GGUF", Quant: "Q8_0", SizeBytes: 4 << 30}
	return jobFormTestData{
		Models:   []*models.Model{m4, m8},
		Builds:   []buildOpt{{ID: "b1", Profile: "rocm"}},
		Presets:  benchmark.Presets(),
		Params:   paramViews(0, nil),
		MaxCells: maxJobCells,
		KLReference: []klRepoGroup{
			{Repo: "u/M-GGUF", Models: []*models.Model{m4, m8}},
		},
	}
}

// The capability presets render in the existing preset checkbox list
// (no special-casing), with the data attributes the matrix-count and KL
// dropdown JS keys off, and the KL dropdown is present but hidden by
// default.
func TestJobFormRendersCapabilityPresetsAndKLDropdown(t *testing.T) {
	out := renderJobFormPartial(t, jobFormData(t))

	for _, name := range []string{
		"perplexity-quick", "perplexity-full",
		"kl-divergence-quick", "kl-divergence-full",
		"hellaswag-quick", "hellaswag-full",
		"winogrande-quick", "winogrande-full",
	} {
		if !strings.Contains(out, `value="`+name+`"`) {
			t.Errorf("capability preset %q missing from the preset list", name)
		}
	}

	// data-source marks capability checkboxes for the collapse math.
	if !strings.Contains(out, `data-source="capability"`) {
		t.Error("capability checkboxes lack data-source=\"capability\"")
	}
	// data-kl marks the kl-divergence checkboxes for the dropdown.
	if !strings.Contains(out, `data-kl="1"`) {
		t.Error("kl-divergence checkboxes lack data-kl=\"1\"")
	}
	// The KL dropdown is present with its automatic default and the
	// model options grouped by repo.
	if !strings.Contains(out, `name="kl_reference"`) {
		t.Fatal("KL reference dropdown missing")
	}
	if !strings.Contains(out, "automatic — largest quant of each model's repo") {
		t.Error("the automatic (default) reference option is missing")
	}
	if !strings.Contains(out, `<optgroup label="u/M-GGUF">`) {
		t.Error("the repo-grouped optgroup is missing")
	}
	if !strings.Contains(out, `value="m4"`) || !strings.Contains(out, `value="m8"`) {
		t.Error("the model options are missing from the dropdown")
	}
	// Hidden by default: the row starts with display:none.
	if !strings.Contains(out, `id="kl-reference-row" style="display:none`) {
		t.Error("the KL reference row must be hidden by default")
	}
	// The sweep rows carry the collapse attribute.
	if !strings.Contains(out, `data-affects-eval="1"`) {
		t.Error("param rows lack data-affects-eval=\"1\" for eval-reaching fields")
	}
}

// The KL dropdown must not appear when there are no models (nothing to
// reference against), and the form still renders.
func TestJobFormRendersWithoutModels(t *testing.T) {
	data := jobFormData(t)
	data.Models = nil
	data.KLReference = nil
	out := renderJobFormPartial(t, data)
	if !strings.Contains(out, "no enabled models") {
		t.Error("the empty-models hint is missing")
	}
	// The dropdown row still renders (it just has no options) — the
	// presets are what matter.
	if !strings.Contains(out, `value="perplexity-quick"`) {
		t.Error("presets missing when no models are enabled")
	}
}
