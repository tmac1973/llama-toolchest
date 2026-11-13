package api

import (
	"bytes"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The CPU share field appears only once the MoE cache is on, with its
// effective flag or the note that the build is too old for it.
func TestMoECacheCPUShareFieldFollowsTheCache(t *testing.T) {
	base := testTemplates(t)
	render := func(data modelConfigPanelData) string {
		data.ModelID = "m"
		data.HasExperts, data.NLayers = true, 48
		data.DraftModes, data.AssistModes = models.DraftModes(), models.AssistModes()
		var buf bytes.Buffer
		if err := base.ExecuteTemplate(&buf, "model_config", data); err != nil {
			t.Fatalf("execute: %v", err)
		}
		return buf.String()
	}

	off := render(modelConfigPanelData{Config: &models.ModelConfig{GPULayers: 999, CPUMoE: 12}})
	if strings.Contains(off, `name="moe_cache_cpu_share"`) {
		t.Error("cache off: the CPU share field is shown")
	}

	on := render(modelConfigPanelData{Config: &models.ModelConfig{GPULayers: 999, CPUMoE: 12, MoECacheMiB: 1024, MoECacheCPUShare: 0.9}})
	for _, want := range []string{`name="moe_cache_cpu_share"`, `value="0.9"`, "--moe-cache-cpu-share 0.9"} {
		if !strings.Contains(on, want) {
			t.Errorf("cache on: panel missing %q", want)
		}
	}

	old := render(modelConfigPanelData{
		Config:               &models.ModelConfig{GPULayers: 999, CPUMoE: 12, MoECacheMiB: 1024, MoECacheCPUShare: 0.9},
		MoECacheCPUShareNote: "older than b12191",
	})
	if !strings.Contains(old, "older than b12191") || strings.Contains(old, "--moe-cache-cpu-share") {
		t.Error("old build: panel shows the flag instead of the note")
	}
}

// The share is saved from 0 to 1, refused outside it, and cleared with
// the cache it belongs to.
func TestConfigFormSavesMoECacheCPUShare(t *testing.T) {
	s := newProfileServer(t, "b100")
	path := "/api/models/" + profTestID + "/config"

	s.doProfileRequest(t, "PUT", path, url.Values{"cpu_moe": {"8"}, "moe_cache_mib": {"1024"}, "moe_cache_cpu_share": {"0.9"}})
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.MoECacheCPUShare != 0.9 {
		t.Fatalf("saved share = %v, want 0.9", cfg.MoECacheCPUShare)
	}

	for _, bad := range []string{"1.5", "-0.1", "NaN"} {
		code, _ := s.doProfileRequest(t, "PUT", path, url.Values{"cpu_moe": {"8"}, "moe_cache_mib": {"1024"}, "moe_cache_cpu_share": {bad}})
		if code != http.StatusBadRequest {
			t.Errorf("share %s: status %d, want 400", bad, code)
		}
	}
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.MoECacheCPUShare != 0.9 {
		t.Errorf("share after refused posts = %v, want 0.9 kept", cfg.MoECacheCPUShare)
	}

	s.doProfileRequest(t, "PUT", path, url.Values{"cpu_moe": {"8"}, "moe_cache_mib": {"0"}, "moe_cache_cpu_share": {"0.9"}})
	if cfg, _ := s.registry.GetConfig(profTestID); cfg.MoECacheCPUShare != 0 {
		t.Errorf("share with the cache off = %v, want 0", cfg.MoECacheCPUShare)
	}
}
