package backup

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/builder"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/monitor"
)

func testState(t *testing.T) (*config.Config, *builder.Builder, *models.Registry) {
	t.Helper()
	dataDir := t.TempDir()
	modelsDir := filepath.Join(dataDir, "models")
	os.MkdirAll(modelsDir, 0o755)

	cfg := &config.Config{
		ListenAddr: ":3000", DataDir: dataDir, LlamaPort: 8080,
		LogLevel: "info", ModelsMax: 2, AutoStart: true,
		HFToken: "hf_secret_token", APIKey: "api_secret_key",
		RuntimeEnv:      map[string]string{"ROCBLAS_USE_HIPBLASLT": "1"},
		RuntimeEnvExtra: "FOO=bar",
	}
	// Config.ModelsPath falls back to <DataDir>/models when ModelsDir is
	// empty, matching production defaults.

	b := builder.NewBuilder(dataDir)
	if err := b.SaveFlagPreset(builder.FlagPreset{
		Name: "rocwmma", Profile: "rocm",
		Options: map[string]bool{"GGML_HIP_ROCWMMA_FATTN": true},
	}); err != nil {
		t.Fatal(err)
	}

	reg := models.NewRegistry(dataDir, modelsDir)
	m := &models.Model{
		ID: "org--repo-GGUF--model-Q4_K_M", ModelID: "org/repo-GGUF",
		Quant: "Q4_K_M", Filename: "model-Q4_K_M.gguf",
		FilePath: filepath.Join(modelsDir, "org--repo-GGUF", "model-Q4_K_M.gguf"),
	}
	if err := reg.Add(m); err != nil {
		t.Fatal(err)
	}
	c, _ := reg.GetConfig(m.ID)
	c.MmprojPath = filepath.Join(modelsDir, "org--repo-GGUF", "mmproj-BF16.gguf")
	c.MtpPath = "/outside/models/mtp.gguf"
	reg.SetConfig(m.ID, c)

	return cfg, b, reg
}

func TestAssembleShape(t *testing.T) {
	cfg, b, reg := testState(t)
	f := Assemble(cfg, b, reg, []monitor.GPUInfo{{Name: "RX 9070 XT"}}, false)

	if f.Version != 1 {
		t.Errorf("version = %d", f.Version)
	}
	if f.Settings == nil || f.Settings.ModelsMax == nil || *f.Settings.ModelsMax != 2 {
		t.Errorf("settings preference fields missing: %+v", f.Settings)
	}
	if f.RuntimeEnv == nil || f.RuntimeEnv.Curated["ROCBLAS_USE_HIPBLASLT"] != "1" || f.RuntimeEnv.Extra != "FOO=bar" {
		t.Errorf("runtime env wrong: %+v", f.RuntimeEnv)
	}
	if len(f.FlagPresets) != 1 || f.FlagPresets[0].Name != "rocwmma" {
		t.Errorf("flag presets wrong: %+v", f.FlagPresets)
	}
	if len(f.ModelConfigs) != 1 {
		t.Fatalf("model configs: %+v", f.ModelConfigs)
	}
	mc := f.ModelConfigs[0]
	if mc.ModelID != "org/repo-GGUF" || mc.Quant != "Q4_K_M" || mc.Filename != "model-Q4_K_M.gguf" {
		t.Errorf("identity wrong: %+v", mc)
	}

	// No registry IDs or absolute model paths in the model_configs section.
	data, _ := json.Marshal(f.ModelConfigs)
	if bytes.Contains(data, []byte("org--repo-GGUF--model")) {
		t.Error("registry ID leaked into export")
	}
	if mc.Config.MmprojPath != filepath.Join("org--repo-GGUF", "mmproj-BF16.gguf") {
		t.Errorf("mmproj not relativized: %q", mc.Config.MmprojPath)
	}
	if mc.Config.MtpPath != "/outside/models/mtp.gguf" {
		t.Errorf("outside-path should export verbatim: %q", mc.Config.MtpPath)
	}
}

func TestAssembleNoSecretsByDefault(t *testing.T) {
	cfg, b, reg := testState(t)

	data, err := Assemble(cfg, b, reg, nil, false).Marshal()
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{"hf_secret_token", "api_secret_key", "hf_token", "api_key"} {
		if bytes.Contains(data, []byte(secret)) {
			t.Errorf("secrets-off export contains %q", secret)
		}
	}

	withSecrets, _ := Assemble(cfg, b, reg, nil, true).Marshal()
	for _, want := range []string{`"hf_token": "hf_secret_token"`, `"api_key": "api_secret_key"`} {
		if !bytes.Contains(withSecrets, []byte(want)) {
			t.Errorf("secrets-on export missing %s", want)
		}
	}
}

// A server with no secrets must not emit empty secret strings even with
// the flag on — a present-but-empty secret could blank a target's
// credential on restore.
func TestAssembleNeverEmitsEmptySecrets(t *testing.T) {
	cfg, b, reg := testState(t)
	cfg.HFToken, cfg.APIKey = "", ""
	data, _ := Assemble(cfg, b, reg, nil, true).Marshal()
	for _, key := range []string{"hf_token", "api_key"} {
		if bytes.Contains(data, []byte(key)) {
			t.Errorf("empty secret emitted under %q", key)
		}
	}
}

func TestAssembleDeterministic(t *testing.T) {
	cfg, b, reg := testState(t)
	a1 := Assemble(cfg, b, reg, nil, false)
	a2 := Assemble(cfg, b, reg, nil, false)
	a1.ExportedAt = time.Time{}
	a2.ExportedAt = time.Time{}
	d1, _ := a1.Marshal()
	d2, _ := a2.Marshal()
	if !bytes.Equal(d1, d2) {
		t.Error("two exports of unchanged state differ")
	}
}

// Duplicate (ModelID, Quant) registrations must collapse to one entry,
// chosen by smallest registry ID, or the export is non-deterministic.
func TestAssembleDedupesIdentity(t *testing.T) {
	cfg, b, reg := testState(t)
	dup := &models.Model{
		ID: "aaa-duplicate-earlier-id", ModelID: "org/repo-GGUF",
		Quant: "Q4_K_M", Filename: "model-Q4_K_M.gguf",
	}
	reg.Add(dup)
	c, _ := reg.GetConfig(dup.ID)
	c.ContextSize = 4242 // distinguishable
	reg.SetConfig(dup.ID, c)

	f := Assemble(cfg, b, reg, nil, false)
	if len(f.ModelConfigs) != 1 {
		t.Fatalf("expected 1 deduped entry, got %d", len(f.ModelConfigs))
	}
	if f.ModelConfigs[0].Config.ContextSize != 4242 {
		t.Errorf("smallest registry ID should win, got ctx=%d", f.ModelConfigs[0].Config.ContextSize)
	}
}
