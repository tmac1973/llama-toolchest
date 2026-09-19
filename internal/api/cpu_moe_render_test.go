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

// The CPU Expert Layers field appears only for mixture-of-experts models,
// with its effective flag, and the system-memory note only when something
// is kept there.
func TestCPUMoEFieldOnlyForMoEModels(t *testing.T) {
	base, err := template.New("").Funcs(testFuncMap).ParseFS(web.Templates,
		"templates/layout.html", "templates/partials/*.html")
	if err != nil {
		t.Fatalf("parse templates: %v", err)
	}
	render := func(data modelConfigPanelData) string {
		data.ModelID = "m"
		data.DraftModes, data.AssistModes = models.DraftModes(), models.AssistModes()
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
			t.Fatalf("execute: %v", err)
		}
		return buf.String()
	}

	dense := render(modelConfigPanelData{Config: &models.ModelConfig{GPULayers: 999}})
	if strings.Contains(dense, `name="cpu_moe"`) {
		t.Error("dense model shows the CPU Expert Layers field")
	}

	moe := render(modelConfigPanelData{
		Config:      &models.ModelConfig{GPULayers: 999, CPUMoE: 12},
		HasExperts:  true,
		NLayers:     48,
		CPURAMLabel: "About 9.5 GiB of the model's weights stay in system memory with these settings.",
	})
	for _, want := range []string{`name="cpu_moe"`, `max="48"`, "--n-cpu-moe 12", "About 9.5 GiB"} {
		if !strings.Contains(moe, want) {
			t.Errorf("MoE panel missing %q", want)
		}
	}
}

// A swept or overridden cpu_moe reaches the config the benchmark launches.
func TestSnapshotCarriesCPUMoEToLaunchConfig(t *testing.T) {
	snap := benchmark.SnapshotFromConfig(models.ModelConfig{CPUMoE: 7}, "", false)
	if got := applySnapshotToConfig(models.ModelConfig{}, snap); got.CPUMoE != 7 {
		t.Errorf("launch config CPUMoE = %d, want 7", got.CPUMoE)
	}
}
