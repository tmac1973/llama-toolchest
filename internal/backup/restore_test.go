package backup

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

// depsRecorder is a fake Deps that records every call.
type depsRecorder struct {
	deps       Deps
	settings   []Settings
	env        []RuntimeEnv
	presets    []builder.FlagPreset
	configs    map[string]models.ModelConfig
	pending    []MissingModel
	installed  map[string][]string // "modelID quant" -> registry IDs
	currentEnv RuntimeEnv
}

func newRecorder(numGPUs int, modelsDir string) *depsRecorder {
	r := &depsRecorder{
		configs:   map[string]models.ModelConfig{},
		installed: map[string][]string{},
	}
	r.deps = Deps{
		ApplySettings: func(s Settings) ([]string, error) {
			r.settings = append(r.settings, s)
			var changed []string
			if s.ModelsMax != nil {
				changed = append(changed, "models_max")
			}
			return changed, nil
		},
		CurrentEnv: func() RuntimeEnv { return r.currentEnv },
		ApplyEnv: func(e RuntimeEnv) error {
			r.env = append(r.env, e)
			return nil
		},
		SaveFlagPreset: func(p builder.FlagPreset) error {
			if !strings.HasPrefix(p.Name, "ok") {
				// mimic real name validation failing
				return errInvalidName
			}
			r.presets = append(r.presets, p)
			return nil
		},
		InstalledModels: func(modelID, quant string) []string {
			return r.installed[modelID+" "+quant]
		},
		ApplyModelConfig: func(id string, cfg models.ModelConfig) error {
			r.configs[id] = cfg
			return nil
		},
		NumGPUs:   numGPUs,
		ModelsDir: modelsDir,
	}
	return r
}

var errInvalidName = errors.New("invalid name: lowercase letters, digits, and hyphens")

func TestParseRejectsWholeFile(t *testing.T) {
	rec := newRecorder(2, "")
	cases := map[string]string{
		"truncated":     `{"version": 1, "settings":`,
		"wrong version": `{"version": 2}`,
		"no identity":   `{"version": 1, "model_configs": [{"quant": "Q4"}]}`,
		"no filename":   `{"version": 1, "model_configs": [{"model_id": "a/b", "quant": "Q4"}]}`,
		"bad preset":    `{"version": 1, "flag_presets": [{"name": "x"}]}`,
	}
	for name, data := range cases {
		f, err := Parse([]byte(data))
		if err == nil {
			t.Errorf("%s: expected rejection", name)
			continue
		}
		if f != nil {
			t.Errorf("%s: rejected parse returned a file", name)
		}
	}
	if len(rec.settings)+len(rec.env)+len(rec.presets)+len(rec.configs) != 0 {
		t.Error("rejected files must apply nothing")
	}
}

func roundTripFile(t *testing.T) *File {
	t.Helper()
	data := []byte(`{
		"version": 1,
		"settings": {"models_max": 3, "auto_start": false},
		"runtime_env": {"curated": {"GGML_CUDA_DISABLE_GRAPHS": "1"}, "extra": "NEW=1"},
		"flag_presets": [{"name": "ok-fast", "profile": "rocm", "options": {"GGML_LTO": true}}],
		"model_configs": [
			{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a-Q4_K_M.gguf",
			 "config": {"enabled": true, "gpu_layers": 999, "context_size": 16384, "threads": 8,
			            "tensor_split": "1,1,0,0", "split_mode": "tensor", "gpu_assign": "tensor-2"}},
			{"model_id": "org/b-GGUF", "quant": "Q8_0", "filename": "b-Q8_0.gguf",
			 "config": {"enabled": true, "gpu_layers": 999, "context_size": 8192, "threads": 8}}
		]
	}`)
	f, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func allSections() Selections {
	return Selections{Settings: true, RuntimeEnv: true, FlagPresets: true, ModelConfigs: true}
}

func TestApplyRoundTrip(t *testing.T) {
	rec := newRecorder(4, t.TempDir())
	rec.installed["org/a-GGUF Q4_K_M"] = []string{"id-a"}
	rec.installed["org/b-GGUF Q8_0"] = []string{"id-b"}
	rec.currentEnv = RuntimeEnv{Curated: map[string]string{"ROCBLAS_USE_HIPBLASLT": "1"}, Extra: "OLD=1"}

	rep := Apply(roundTripFile(t), allSections(), rec.deps)

	if rep.Error != "" || len(rep.Skipped) != 0 {
		t.Fatalf("unexpected failures: %+v", rep)
	}
	if rep.AppliedModelConfigs != 2 || len(rec.configs) != 2 {
		t.Errorf("model configs applied = %d", rep.AppliedModelConfigs)
	}
	if len(rec.presets) != 1 || rec.presets[0].Name != "ok-fast" {
		t.Errorf("preset not applied: %+v", rec.presets)
	}
	// Env merged: target-only key survives, file key added, extra replaced.
	if len(rec.env) != 1 {
		t.Fatalf("env applies = %d", len(rec.env))
	}
	merged := rec.env[0]
	if merged.Curated["ROCBLAS_USE_HIPBLASLT"] != "1" || merged.Curated["GGML_CUDA_DISABLE_GRAPHS"] != "1" {
		t.Errorf("env merge wrong: %+v", merged.Curated)
	}
	if merged.Extra != "NEW=1" {
		t.Errorf("non-empty file extra should replace: %q", merged.Extra)
	}
	// Restart reminder present.
	if len(rep.Notes) == 0 || !strings.Contains(rep.Notes[0], "next server restart") {
		t.Errorf("missing restart note: %+v", rep.Notes)
	}
}

// A file whose extra is empty must not clear the target's extra.
func TestApplyEnvNeverDeletes(t *testing.T) {
	rec := newRecorder(2, "")
	rec.currentEnv = RuntimeEnv{Curated: map[string]string{}, Extra: "KEEP=1"}
	f, _ := Parse([]byte(`{"version": 1, "runtime_env": {"curated": {"GGML_CUDA_DISABLE_GRAPHS": "1"}}}`))
	Apply(f, Selections{RuntimeEnv: true}, rec.deps)
	if rec.env[0].Extra != "KEEP=1" {
		t.Errorf("target extra deleted: %q", rec.env[0].Extra)
	}
}

func TestApplyPerItemFailure(t *testing.T) {
	rec := newRecorder(2, "")
	f, _ := Parse([]byte(`{"version": 1, "flag_presets": [
		{"name": "ok-good", "profile": "rocm"},
		{"name": "BAD NAME", "profile": "rocm"},
		{"name": "ok-more", "profile": "cuda"}
	]}`))
	rep := Apply(f, Selections{FlagPresets: true}, rec.deps)
	if len(rec.presets) != 2 {
		t.Errorf("valid presets should apply, got %d", len(rec.presets))
	}
	if len(rep.Skipped) != 1 || !strings.Contains(rep.Skipped[0].Item, "BAD NAME") {
		t.Errorf("invalid preset should skip with reason: %+v", rep.Skipped)
	}
}

func TestSelectionsFilter(t *testing.T) {
	rec := newRecorder(4, "")
	rep := Apply(roundTripFile(t), Selections{FlagPresets: true}, rec.deps)
	if len(rec.settings)+len(rec.env)+len(rec.configs) != 0 {
		t.Error("unselected sections must not touch deps")
	}
	joined := strings.Join(rep.NotSelected, ",")
	for _, want := range []string{"settings", "runtime env", "model configs"} {
		if !strings.Contains(joined, want) {
			t.Errorf("NotSelected missing %q: %v", want, rep.NotSelected)
		}
	}
}

func TestGPUReResolution(t *testing.T) {
	rec := newRecorder(2, t.TempDir())
	rec.installed["org/a-GGUF Q4_K_M"] = []string{"id-a"}
	f, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a.gguf",
		 "config": {"gpu_assign": "tensor-4", "tensor_split": "1,1,1,1", "split_mode": "tensor"}}
	]}`))
	rep := Apply(f, Selections{ModelConfigs: true}, rec.deps)

	got := rec.configs["id-a"]
	if got.GPUAssign != "all" || got.TensorSplit != "" || got.SplitMode != "layer" {
		t.Errorf("out-of-range fallback wrong: %+v", got)
	}
	found := false
	for _, w := range rep.Warnings {
		if strings.Contains(w, "tensor-4") {
			found = true
		}
	}
	if !found {
		t.Errorf("fallback should warn naming the original: %v", rep.Warnings)
	}

	// In-range re-resolution to the local padded split.
	rec2 := newRecorder(4, t.TempDir())
	rec2.installed["org/a-GGUF Q4_K_M"] = []string{"id-a"}
	f2, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a.gguf",
		 "config": {"gpu_assign": "0-1", "tensor_split": "1,1", "split_mode": "layer"}}
	]}`))
	Apply(f2, Selections{ModelConfigs: true}, rec2.deps)
	if got := rec2.configs["id-a"]; got.TensorSplit != "1,1,0,0" {
		t.Errorf("in-range re-resolution wrong: %+v", got)
	}
}

func TestPathResolution(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "org--a-GGUF"), 0o755)
	os.WriteFile(filepath.Join(dir, "org--a-GGUF", "mmproj.gguf"), []byte("x"), 0o644)

	rec := newRecorder(1, dir)
	rec.installed["org/a-GGUF Q4_K_M"] = []string{"id-a"}
	f, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a.gguf",
		 "config": {"mmproj_path": "org--a-GGUF/mmproj.gguf", "mtp_path": "org--a-GGUF/missing.gguf"}}
	]}`))
	rep := Apply(f, Selections{ModelConfigs: true}, rec.deps)

	got := rec.configs["id-a"]
	if got.MmprojPath != filepath.Join(dir, "org--a-GGUF", "mmproj.gguf") {
		t.Errorf("existing relative path should resolve absolute: %q", got.MmprojPath)
	}
	if got.MtpPath != "" {
		t.Errorf("missing path should blank: %q", got.MtpPath)
	}
	if len(rep.Warnings) == 0 {
		t.Error("blanked path should warn")
	}
}

func TestMissingModelWithoutPending(t *testing.T) {
	rec := newRecorder(2, "")
	f, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/x-GGUF", "quant": "Q4_K_M", "filename": "x.gguf", "config": {}}
	]}`))
	rep := Apply(f, Selections{ModelConfigs: true}, rec.deps)
	if len(rep.Missing) != 1 || rep.Missing[0].Pending {
		t.Errorf("missing entry wrong: %+v", rep.Missing)
	}
	if len(rep.Skipped) != 1 || rep.Skipped[0].Reason != "model not installed" {
		t.Errorf("nil SavePending should skip: %+v", rep.Skipped)
	}
}

// A file containing one model config leaves other installed models'
// configs untouched — merge never deletes.
func TestMergeNeverDeletes(t *testing.T) {
	rec := newRecorder(2, "")
	rec.installed["org/a-GGUF Q4_K_M"] = []string{"id-a"}
	rec.installed["org/other Q8_0"] = []string{"id-other"}
	f, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a.gguf", "config": {}}
	]}`))
	Apply(f, Selections{ModelConfigs: true}, rec.deps)
	if _, touched := rec.configs["id-other"]; touched {
		t.Error("restore touched a model absent from the file")
	}
}

// Duplicate registrations of one identity all receive the config.
func TestDuplicateIdentityAppliesToAll(t *testing.T) {
	rec := newRecorder(2, "")
	rec.installed["org/a-GGUF Q4_K_M"] = []string{"id-1", "id-2"}
	f, _ := Parse([]byte(`{"version": 1, "model_configs": [
		{"model_id": "org/a-GGUF", "quant": "Q4_K_M", "filename": "a.gguf", "config": {}}
	]}`))
	rep := Apply(f, Selections{ModelConfigs: true}, rec.deps)
	if rep.AppliedModelConfigs != 2 || len(rec.configs) != 2 {
		t.Errorf("both duplicates should apply, got %d", rep.AppliedModelConfigs)
	}
}

// Round trip against the real Assemble output.
func TestAssembleParseApplyRoundTrip(t *testing.T) {
	cfg, b, reg := testState(t)
	data, err := Assemble(cfg, b, reg, []monitor.GPUInfo{{Name: "gpu"}}, true).Marshal()
	if err != nil {
		t.Fatal(err)
	}
	f, err := Parse(data)
	if err != nil {
		t.Fatalf("assembled file must parse: %v", err)
	}
	rec := newRecorder(1, t.TempDir())
	rec.installed["org/repo-GGUF Q4_K_M"] = []string{"target-id"}
	// Rename the assembled preset so the recorder's fake validator accepts it.
	f.FlagPresets[0].Name = "ok-" + f.FlagPresets[0].Name
	rep := Apply(f, allSections(), rec.deps)
	if len(rep.Skipped) != 0 {
		t.Errorf("round trip skipped items: %+v", rep.Skipped)
	}
	if rep.AppliedModelConfigs != 1 {
		t.Errorf("model config did not land: %+v", rep)
	}
}
