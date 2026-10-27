package api

import (
	"bytes"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
)

// The web UI link uses the external URL's host and the port as seen from
// outside, which a container may publish elsewhere.
func TestGenWebUIURL(t *testing.T) {
	s := vramServer(t)
	s.cfg.ExternalURL = "https://compute2:3000"
	if got := s.genWebUIURL(8082); got != "https://compute2:8082" {
		t.Errorf("same ports: %q", got)
	}
	s.cfg.GenPublicPortStart = 9081
	if got := s.genWebUIURL(8082); got != "https://compute2:9082" {
		t.Errorf("published elsewhere: %q, want port 9082", got)
	}
	s.cfg.ExternalURL = "not a url"
	if got := s.genWebUIURL(8082); got != "" {
		t.Errorf("no host: %q", got)
	}
}

// In a container the link is only offered when setup.sh published the
// ports; otherwise it would lead nowhere.
func TestGenWebUIURLNeedsPublishedPortsInContainer(t *testing.T) {
	s := vramServer(t)
	s.cfg.ExternalURL = "https://compute2:3000"
	s.inContainerFn = func() bool { return true }
	s.cfg.GenPortsPublished = false
	if got := s.genWebUIURL(8082); got != "" {
		t.Errorf("unpublished ports: %q, want no link", got)
	}
	s.cfg.GenPortsPublished = true
	if got := s.genWebUIURL(8082); got != "https://compute2:8082" {
		t.Errorf("published ports: %q", got)
	}
}

func TestBannerGPUs(t *testing.T) {
	gpus := []models.GPUSpec{{Index: 0, Name: "R9700"}, {Index: 1, Name: "A4000"}}
	running := func(g int) bool { return g == 0 }
	if got := bannerGPUs(gpus, running, map[int]int{0: 8000}); len(got) != 1 || got[0].Index != 0 || got[0].Name != "R9700" {
		t.Errorf("job and LLM on GPU 0: %+v", got)
	}
	if got := bannerGPUs(gpus, running, map[int]int{1: 8000}); len(got) != 0 {
		t.Errorf("job on GPU 0, LLM on GPU 1: %+v, want no banner", got)
	}
	if got := bannerGPUs(gpus, func(int) bool { return false }, map[int]int{0: 8000}); len(got) != 0 {
		t.Errorf("no job: %+v", got)
	}
}

func TestBannerPartialRenders(t *testing.T) {
	s := newTestServer(t)
	var buf bytes.Buffer
	if err := s.pages["server.html"].ExecuteTemplate(&buf, "gen_banner", []genBannerGPU{{Index: 0, Name: "R9700"}}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "Chat is slower while image or music generation runs on GPU 0 (R9700)") {
		t.Errorf("banner = %s", buf.String())
	}
	// No job running: nothing renders.
	rec := httptest.NewRecorder()
	s.cfg.GenerationEnabled = true
	s.handleGenBanner(rec, httptest.NewRequest("GET", "/api/gen/banner", nil))
	if strings.TrimSpace(rec.Body.String()) != "" {
		t.Errorf("banner with nothing running: %q", rec.Body.String())
	}
}

// The table links a running instance's web UI and shows its GPU's jobs.
func TestInstanceTableWebUIAndJobs(t *testing.T) {
	s := vramServer(t)
	s.cfg.ExternalURL = "http://asmodean:3000"
	fb := newFakeBackend(t, 0)
	// The build in instanceServer's builds.json has no web UI.
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	tk, _ := s.genQueue.Submit(0, "ming", generation.KindSD)
	defer tk.Done()
	s.genQueue.Submit(0, "ming", generation.KindSD)

	rows := s.genInstanceRows(s.gen)
	if len(rows) != 1 || rows[0].GPUJobsRunning != 1 || rows[0].GPUJobsWaiting != 1 {
		t.Fatalf("rows = %+v", rows)
	}
	if rows[0].WebUIURL != "" {
		t.Error("a web UI link for a build without one")
	}

	var buf bytes.Buffer
	rows[0].WebUIURL = "http://asmodean:8081"
	if err := s.pages["server.html"].ExecuteTemplate(&buf, "gen_instances_table", genInstancesData{Rows: rows}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Open web UI →", "skip the llama-toolchest queue", "1 running, 1 waiting"} {
		if !strings.Contains(buf.String(), want) {
			t.Errorf("table lacks %q", want)
		}
	}
}
