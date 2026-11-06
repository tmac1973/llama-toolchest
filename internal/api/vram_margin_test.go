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
	if s.cfg.VRAMMarginAll() != 1536 || s.cfg.VRAMMarginByGPU[0] != 2048 {
		t.Errorf("saved %d and %v, want 1536 and GPU 0 at 2048", s.cfg.VRAMMarginAll(), s.cfg.VRAMMarginByGPU)
	}
	if strings.Contains(rec.Body.String(), "Below 1 GiB") {
		t.Errorf("margins of 1 GiB and more warn: %s", rec.Body)
	}
	if _, ok := s.cfg.VRAMMarginByGPU[1]; ok {
		t.Error("a blank GPU row saved a margin")
	}
	// GPU 3 is not on the form, so it keeps its margin.
	if s.cfg.VRAMMarginByGPU[3] != 1024 {
		t.Errorf("GPU 3 = %d, want its saved 1024", s.cfg.VRAMMarginByGPU[3])
	}

	// Blank clears a GPU's own margin; blank for every GPU is the default.
	putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {""}, "vram_margin_gpu_0": {""}, "vram_margin_gpu_3": {""}})
	if s.cfg.VRAMMarginMiB != nil || s.cfg.VRAMMarginByGPU != nil {
		t.Errorf("saved %v and %v, want the default and none", s.cfg.VRAMMarginMiB, s.cfg.VRAMMarginByGPU)
	}

	// 0 is 0 MiB, not the default, and below 1 GiB the save warns.
	for gib, want := range map[string]int{"0": 0, "0.5": 512} {
		rec := putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {gib}})
		if rec.Code != http.StatusOK || s.cfg.VRAMMarginAll() != want {
			t.Errorf("%q: status %d, saved %d MiB, want %d", gib, rec.Code, s.cfg.VRAMMarginAll(), want)
		}
		if !strings.Contains(rec.Body.String(), "Below 1 GiB") {
			t.Errorf("%q: the save does not warn: %s", gib, rec.Body)
		}
	}
	rec = putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {"2"}, "vram_margin_gpu_0": {"0"}})
	if rec.Code != http.StatusOK || s.cfg.VRAMMarginFor(0) != 0 || !strings.Contains(rec.Body.String(), "Below 1 GiB") {
		t.Errorf("GPU 0 at 0: status %d, margin %d: %s", rec.Code, s.cfg.VRAMMarginFor(0), rec.Body)
	}

	for _, bad := range []string{"-1", "abc", "2000"} {
		rec := putMarginForm(t, s, url.Values{"vram_margin_touched": {"1"}, "vram_margin_gib": {bad}})
		if rec.Code != http.StatusBadRequest {
			t.Errorf("%q: status %d, want 400", bad, rec.Code)
		}
	}
	if s.cfg.VRAMMarginAll() != 2048 {
		t.Errorf("a refused value was saved: %d", s.cfg.VRAMMarginAll())
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
	if s.cfg.VRAMMarginAll() != 1024 || s.cfg.VRAMMarginByGPU[1] != 3072 {
		t.Errorf("saved %d and %v", s.cfg.VRAMMarginAll(), s.cfg.VRAMMarginByGPU)
	}
	if code := put(`{"vram_margin_by_gpu": {}}`); code != http.StatusOK || s.cfg.VRAMMarginByGPU != nil {
		t.Errorf("status %d, margins %v; want {} to clear them", code, s.cfg.VRAMMarginByGPU)
	}
	if code := put(`{"vram_margin_mib": -5}`); code != http.StatusBadRequest {
		t.Errorf("negative margin: status %d, want 400", code)
	}
	// 0 is 0 MiB.
	if code := put(`{"vram_margin_mib": 0, "vram_margin_by_gpu": {"1": 0}}`); code != http.StatusOK ||
		s.cfg.VRAMMarginFor(0) != 0 || s.cfg.VRAMMarginFor(1) != 0 {
		t.Errorf("status %d, margins %d and %d; want 0 MiB", code, s.cfg.VRAMMarginFor(0), s.cfg.VRAMMarginFor(1))
	}
}

// Each card gets its own margin, or the margin for every GPU; the router
// gets the largest, but never 0. The default is 1 GiB.
func TestHardwareAppliesVRAMMargin(t *testing.T) {
	s := newTestServer(t)
	s.testHardware = &models.Hardware{GPUs: []models.GPUSpec{
		{Index: 0, Name: "A", VRAMTotalMiB: 24576},
		{Index: 1, Name: "B", VRAMTotalMiB: 16384},
	}}
	if got := s.routerMemoryMarginMiB(); got != config.DefaultVRAMMarginMiB {
		t.Errorf("router margin with the default = %d MiB, want 1 GiB", got)
	}
	s.cfg.VRAMMarginMiB = ptr(0)
	if gpus := s.hardware().GPUs; gpus[0].MarginMiB != 0 || models.CardBudgetMiB(gpus[0]) != 24576 {
		t.Errorf("margin 0: card margin %d, budget %d MiB; want nothing kept free", gpus[0].MarginMiB, models.CardBudgetMiB(gpus[0]))
	}
	if got := s.routerMemoryMarginMiB(); got != 1 {
		t.Errorf("router margin with every margin 0 = %d MiB, want 1: 0 turns its accounting off", got)
	}
	s.cfg.VRAMMarginMiB = ptr(1536)
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
	s.cfg.VRAMMarginMiB = nil
	s.cfg.VRAMMarginByGPU = map[int]int{0: 1536}
	card := func() string {
		rec := httptest.NewRecorder()
		s.handleVRAMMarginCard(rec, httptest.NewRequest(http.MethodGet, "/api/settings/vram-margin", nil))
		return rec.Body.String()
	}
	page := card()
	for _, want := range []string{
		`name="vram_margin_touched"`,
		`name="vram_margin_gib" value="1"`, // the default, 1 GiB
		`name="vram_margin_gpu_0" value="1.5"`,
		`name="vram_margin_gpu_1" value=""`,
		"0: RX 9070 XT", "1: R9700",
	} {
		if !strings.Contains(page, want) {
			t.Errorf("card lacks %q:\n%s", want, page)
		}
	}
	if strings.Contains(page, "Below 1 GiB") {
		t.Errorf("the card warns with no margin below 1 GiB:\n%s", page)
	}
	s.cfg.VRAMMarginByGPU = map[int]int{1: 0}
	page = card()
	for _, want := range []string{`name="vram_margin_gpu_1" value="0"`, "Below 1 GiB"} {
		if !strings.Contains(page, want) {
			t.Errorf("GPU 1 at 0: card lacks %q:\n%s", want, page)
		}
	}
}

// A restore sets the margin and merges the per-GPU margins: a GPU the
// backup does not name keeps its own.
func TestRestoreVRAMMargin(t *testing.T) {
	s := newSettingsServer(t, &config.Config{VRAMMarginByGPU: map[int]int{2: 4096}})
	changed, err := s.restoreDeps().ApplySettings(backup.Settings{VRAMMarginMiB: ptr(1280),
		VRAMMarginByGPU: map[int]int{1: 2048, 0: 1024, 3: 0}, VRAMMarginZeroOK: true})
	if err != nil {
		t.Fatal(err)
	}
	if s.cfg.VRAMMarginAll() != 1280 || s.cfg.VRAMMarginByGPU[0] != 1024 || s.cfg.VRAMMarginByGPU[1] != 2048 ||
		s.cfg.VRAMMarginByGPU[2] != 4096 || s.cfg.VRAMMarginFor(3) != 0 {
		t.Errorf("after restore: %d and %v", s.cfg.VRAMMarginAll(), s.cfg.VRAMMarginByGPU)
	}
	want := []string{"vram_margin_mib", "vram_margin_by_gpu[0]", "vram_margin_by_gpu[1]", "vram_margin_by_gpu[3]"}
	if strings.Join(changed, ",") != strings.Join(want, ",") {
		t.Errorf("changed = %v, want %v", changed, want)
	}
}

// A backup from before 0 meant 0 MiB: 0 there was the default, and any
// value below 1 GiB counted as 1 GiB, so restore keeps that.
func TestRestoreOldVRAMMargin(t *testing.T) {
	s := newSettingsServer(t, &config.Config{VRAMMarginMiB: ptr(2048)})
	if _, err := s.restoreDeps().ApplySettings(backup.Settings{VRAMMarginMiB: ptr(0), VRAMMarginByGPU: map[int]int{0: 0, 1: 512, 2: 3072}}); err != nil {
		t.Fatal(err)
	}
	if s.cfg.VRAMMarginAll() != 1024 || s.cfg.VRAMMarginFor(0) != 1024 || s.cfg.VRAMMarginFor(1) != 1024 || s.cfg.VRAMMarginFor(2) != 3072 {
		t.Errorf("after restore: %d and %v, want 1 GiB for the zeros and 512", s.cfg.VRAMMarginAll(), s.cfg.VRAMMarginByGPU)
	}
}
