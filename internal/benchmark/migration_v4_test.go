package benchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A v3 file's size_gb / vram_total_mb (values always were binary units)
// must load into SizeGiB / VRAMTotalMiB and be rewritten under the new
// names, with the legacy keys gone.
func TestLoadMigratesLegacyUnitFields(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config", "benchmarks.json")
	os.MkdirAll(filepath.Dir(path), 0o755)
	v3 := `{
  "version": 3,
  "jobs": [{"id": "adhoc", "name": "Ad-Hoc Runs"}],
  "runs": [{
    "id": "r1", "job_id": "adhoc", "status": "completed",
    "model_id": "m", "model_name": "m", "quant": "Q4",
    "size_gb": 16.5,
    "config": {"gpu_layers": 99, "context_size": 4096, "flash_attention": true, "threads": 8},
    "build_id": "", "build_ref": "", "build_profile": "",
    "gpus": [{"index": 0, "name": "RX 7900", "vram_total_mb": 24560}],
    "preset": "internal-standard", "prompt_tokens": [128], "gen_tokens": 128
  }]
}`
	if err := os.WriteFile(path, []byte(v3), 0o644); err != nil {
		t.Fatal(err)
	}

	s := NewStore(dir, nil)
	run, err := s.Get("r1")
	if err != nil {
		t.Fatal(err)
	}
	if run.SizeGiB != 16.5 {
		t.Errorf("SizeGiB = %v, want 16.5", run.SizeGiB)
	}
	if run.LegacySizeGB != 0 {
		t.Errorf("LegacySizeGB = %v, want 0 after migration", run.LegacySizeGB)
	}
	if len(run.GPUs) != 1 || run.GPUs[0].VRAMTotalMiB != 24560 {
		t.Errorf("GPUs = %+v, want VRAMTotalMiB 24560", run.GPUs)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), `"size_gb"`) || strings.Contains(string(data), `"vram_total_mb"`) {
		t.Error("rewritten file still contains legacy field names")
	}
	if !strings.Contains(string(data), `"size_gib": 16.5`) {
		t.Error("rewritten file missing size_gib")
	}
	if !strings.Contains(string(data), `"vram_total_mib": 24560`) {
		t.Error("rewritten file missing vram_total_mib")
	}
}
