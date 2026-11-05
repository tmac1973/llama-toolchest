package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/backup"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// putMarginForm sends the Settings form fields in v, as htmx does.
func putMarginForm(t *testing.T, s *Server, v url.Values) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(v.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("HX-Request", "true")
	rec := httptest.NewRecorder()
	s.handleUpdateSettings(rec, req)
	return rec
}

func TestVRAMMarginForm(t *testing.T) {
	s := newSettingsServer(t, &config.Config{VRAMMarginByGPU: map[int]int{3: 1024}})
	rec := putMarginForm(t, s, url.Values{
		"vram_margin_touched": {"1"},
		"vram_margin_gib":     {"1.5"},
		"vram_margin_gpu_0":   {"2"},
		"vram_margin_gpu_1":   {""},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if s.cfg.VRAMMarginMiB != 1536 || s.cfg.VRAMMarginByGPU[0] != 2048 {
		t.Errorf("saved %d and %v, want 1536 and GPU 0 at 2048", s.cfg.VRAMMarginMiB, s.cfg.VRAMMarginByGPU)
	}
	if _, ok := s.cfg.VRAMMarginByGPU[1]; ok {
		t.Error("a blank GPU row saved a margin")
	}
	// GPU 3 is not on the form, so it keeps its margin.
	if s.cfg.VRAMMarginByGPU[3] != 1024 {
		t.Errorf("GPU 3 = %d, want its saved 1024", s.cfg.VRAMMarginByGPU[3])
	}

	// Blank clears a GPU's own margin; blank for every GPU is 0.
	putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {""}, "vram_margin_gpu_0": {""}, "vram_margin_gpu_3": {""}})
	if s.cfg.VRAMMarginMiB != 0 || s.cfg.VRAMMarginByGPU != nil {
		t.Errorf("saved %d and %v, want 0 and none", s.cfg.VRAMMarginMiB, s.cfg.VRAMMarginByGPU)
	}

	for _, bad := range []string{"-1", "0", "0.5", "abc", "2000"} {
		rec := putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {bad}})
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%q: status %d, want 400", bad, rec.Code)
		}
	}
	if s.cfg.VRAMMarginMiB != 0 {
		t.Errorf("a refused value was saved: %d", s.cfg.VRAMMarginMiB)
	}
}

func TestVRAMMarginJSON(t *testing.T) {
	s := newSettingsServer(t, &config.Config{})
	put := func(body string) int {
		req := httptest.NewRequest(http.MethodPut, "/api/settings", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.handleUpdateSettings(rec, req)
		return rec.Code
	}
	if code := put(`{"vram_margin_mib": 1024, "vram_margin_by_gpu": {"1": 3072}}`); code != http.StatusOK {
		t.Fatalf("status %d", code)
	}
	if s.cfg.VRAMMarginMiB != 1024 || s.cfg.VRAMMarginByGPU[1] != 3072 {
		t.Errorf("saved %d and %v", s.cfg.VRAMMarginMiB, s.cfg.VRAMMarginByGPU)
	}
	if code := put(`{"vram_margin_by_gpu": {}}`); code != http.StatusOK || s.cfg.VRAMMarginByGPU != nil {
		t.Errorf("status %d, margins %v; want {} to clear them", code, s.cfg.VRAMMarginByGPU)
	}
	if code := put(`{"vram_margin_mib": -5}`); code != http.StatusBadRequest {
		t.Errorf("negative margin: status %d, want 400", code)
	}
}

// Each card gets its own margin, or the margin for every GPU; the router
// gets the largest, and the default is the 1 GiB minimum.
func TestHardwareAppliesVRAMMargin(t *testing.T) {
	s := newTestServer(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, Name: "A", VRAMTotalMiB: 24576},
		{Index: 1, Name: "B", VRAMTotalMiB: 16384},
	}}
	if got := s.routerMemoryMarginMiB(); got != models.WorkingMemoryMiB {
		t.Errorf("router margin with the default = %d MiB, want the 1 GiB minimum", got)
	}
	s.cfg.VRAMMarginMiB = 1536
	s.cfg.VRAMMarginByGPU = map[int]int{1: 2048}
	gpus := s.hardware().GPUs
	if gpus[0].MarginMiB != 1536 || gpus[1].MarginMiB != 2048 {
		t.Errorf("margins = %d and %d, want 1536 and 2048", gpus[0].MarginMiB, gpus[1].MarginMiB)
	}
	if got := models.CardBudgetMiB(gpus[0]); got != 24576-1536 {
		t.Errorf("budget = %d MiB, want the card less its margin", got)
	}
	if got := s.routerMemoryMarginMiB(); got != 2048 {
		t.Errorf("router margin = %d MiB, want the largest, 2048", got)
	}
}

func TestVRAMMarginCard(t *testing.T) {
	s := newTestServer(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, Name: "RX 9070 XT", VRAMTotalMiB: 16304},
		{Index: 1, Name: "R9700", VRAMTotalMiB: 32624},
	}}
	s.cfg.VRAMMarginMiB = 0
	s.cfg.VRAMMarginByGPU = map[int]int{0: 1536}
	rec := httptest.NewRecorder()
	s.handleVRAMMarginCard(rec, httptest.NewRequest(http.MethodGet, "/api/settings/vram-margin", nil))
	page := rec.Body.String()
	for _, want := range []string{
		`name="vram_margin_touched"`,
		`name="vram_margin_gib" value="1"`, // the default shows as the 1 GiB minimum
		`name="vram_margin_gpu_0" value="1.5"`,
		`name="vram_margin_gpu_1" value=""`,
		"0: RX 9070 XT", "1: R9700",
	} {
		if !strings.Contains(page, want) {
			t.Errorf("card lacks %q:\n%s", want, page)
		}
	}
}

// A restore sets the margin and merges the per-GPU margins: a GPU the
// backup does not name keeps its own.
func TestRestoreVRAMMargin(t *testing.T) {
	s := newSettingsServer(t, &config.Config{VRAMMarginByGPU: map[int]int{2: 4096}})
	all := 1280
	changed, err := s.restoreDeps().ApplySettings(backup.Settings{VRAMMarginMiB: &all, VRAMMarginByGPU: map[int]int{1: 2048, 0: 1024}})
	if err != nil {
		t.Fatal(err)
	}
	if s.cfg.VRAMMarginMiB != 1280 || s.cfg.VRAMMarginByGPU[0] != 1024 || s.cfg.VRAMMarginByGPU[1] != 2048 || s.cfg.VRAMMarginByGPU[2] != 4096 {
		t.Errorf("after restore: %d and %v", s.cfg.VRAMMarginMiB, s.cfg.VRAMMarginByGPU)
	}
	want := []string{"vram_margin_mib", "vram_margin_by_gpu[0]", "vram_margin_by_gpu[1]"}
	if strings.Join(changed, ",") != strings.Join(want, ",") {
		t.Errorf("changed = %v, want %v", changed, want)
	}
}
