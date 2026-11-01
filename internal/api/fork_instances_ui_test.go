package api

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/forks"
)

func renderServerPage(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleServerPage(rec, httptest.NewRequest("GET", "/", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("server page: %d", rec.Code)
	}
	return rec.Body.String()
}

func TestServerPageForkCard(t *testing.T) {
	s, _ := forkInstanceServer(t)
	s.cfg.ForksEnabled = false
	if page := renderServerPage(t, s); strings.Contains(page, `id="fork-instances"`) || strings.Contains(page, "fork-form-dialog") {
		t.Error("fork card shown with forks off")
	}
	s.cfg.ForksEnabled = true
	page := renderServerPage(t, s)
	for _, want := range []string{`id="fork-instances"`, `hx-get="/api/forks/instances"`, `id="fork-form-dialog"`, `id="fork-log-dialog"`, "Add instance"} {
		if !strings.Contains(page, want) {
			t.Errorf("server page lacks %s", want)
		}
	}
}

func forkTable(t *testing.T, s *Server) string {
	t.Helper()
	rec := httptest.NewRecorder()
	s.handleForkInstances(rec, htmxReq("GET", "/api/forks/instances", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("table: %d %s", rec.Code, rec.Body)
	}
	return rec.Body.String()
}

func TestForkInstancesTable(t *testing.T) {
	s, port := forkInstanceServer(t)
	if body := forkTable(t, s); !strings.Contains(body, "No fork instances yet") {
		t.Errorf("empty table:\n%s", body)
	}
	s.forks.AddInstance(forks.Instance{ID: "avalon-qwen", ForkID: "avalon", GPU: 1, Port: port, AutoStart: true})
	body := forkTable(t, s)
	for _, want := range []string{"avalon-qwen", "Avalon", "auto-start", "avalon-ad2c6e7-rocm", "stopped", `hx-post="/api/forks/instances/avalon-qwen/start"`, "Edit", "Delete"} {
		if !strings.Contains(body, want) {
			t.Errorf("stopped row lacks %q", want)
		}
	}
	// The test build is not the pinned commit.
	if !strings.Contains(body, "untested") {
		t.Error("untested build not marked")
	}

	// Started from the card: the table comes back with the start's
	// warnings, Stop instead of Start, and no Edit or Delete.
	s.testForkHostInfo = &forks.HostInfo{RAMGiB: 16}
	rec := httptest.NewRecorder()
	s.handleForkInstanceAction(rec, withIDAction(htmxReq("POST", "/", nil), "avalon-qwen", "start"))
	if rec.Code != http.StatusOK {
		t.Fatalf("start: %d %s", rec.Code, rec.Body)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	s.forkMgr.WaitRunning(ctx, "avalon-qwen")
	body = forkTable(t, s)
	for _, want := range []string{"16 GiB of RAM", `hx-post="/api/forks/instances/avalon-qwen/stop"`} {
		if !strings.Contains(body, want) {
			t.Errorf("running row lacks %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, "/start\"") || strings.Contains(body, "Edit</button>") {
		t.Error("running row still offers Start or Edit")
	}

	// Stopping clears the warnings.
	rec = httptest.NewRecorder()
	s.handleForkInstanceAction(rec, withIDAction(htmxReq("POST", "/", nil), "avalon-qwen", "stop"))
	if strings.Contains(forkTable(t, s), "16 GiB of RAM") {
		t.Error("warnings kept after stop")
	}
}

func TestForkInstanceForm(t *testing.T) {
	s, port := forkInstanceServer(t)
	get := func(target string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		s.handleForkInstanceForm(rec, htmxReq("GET", target, nil))
		return rec
	}
	add := get("/api/forks/instances/form")
	for _, want := range []string{`hx-post="/api/forks/instances"`, `name="id"`, `value="avalon"`, "avalon-ad2c6e7-rocm (untested)", `name="extra_env"`, `id="fork-needs"`} {
		if !strings.Contains(add.Body.String(), want) {
			t.Errorf("add form lacks %q", want)
		}
	}

	s.forks.AddInstance(forks.Instance{ID: "q", ForkID: "avalon", GPU: 1, Port: port,
		ExtraArgs: []string{"-c", "32768", "--alias", "two words"}, ExtraEnv: []string{"A=1", "B=2"}, AutoStart: true})
	edit := get("/api/forks/instances/form?id=q").Body.String()
	for _, want := range []string{`hx-put="/api/forks/instances/q"`, `value="q" disabled`, `-c 32768 --alias &#39;two words&#39;`, "A=1\nB=2", "checked", "Save changes"} {
		if !strings.Contains(edit, want) {
			t.Errorf("edit form lacks %q:\n%s", want, edit)
		}
	}
	// The saved arguments read back the same through the form.
	form := url.Values{"fork": {"avalon"}, "gpu": {"1"}, "extra_args": {"-c 32768 --alias 'two words'"}, "extra_env": {"A=1\nB=2"}}
	rec := httptest.NewRecorder()
	s.handleForkEditInstance(rec, withID(forkForm("PUT", "/", form), "q"))
	if got, _ := s.forks.Instance("q"); rec.Code != http.StatusOK || len(got.ExtraArgs) != 4 || got.ExtraArgs[3] != "two words" {
		t.Errorf("edit round trip: %d %+v", rec.Code, got.ExtraArgs)
	}

	if rec := get("/api/forks/instances/form?id=nope"); rec.Code != http.StatusNotFound {
		t.Errorf("form for an unknown instance: %d", rec.Code)
	}
}

func TestForkNeedsBox(t *testing.T) {
	s, _ := forkInstanceServer(t)
	get := func() string {
		rec := httptest.NewRecorder()
		s.handleForkNeeds(rec, htmxReq("GET", "/api/forks/needs?fork=avalon&gpu=1", nil))
		return rec.Body.String()
	}
	s.testForkHostInfo = &forks.HostInfo{RAMGiB: 128, AVX2Known: true, AVX2: true, GPUArch: "gfx1100"}
	if body := strings.TrimSpace(get()); body != "" {
		t.Errorf("a host that meets all needs got:\n%s", body)
	}
	s.testForkHostInfo = &forks.HostInfo{RAMGiB: 16, GPUArch: "gfx1201"}
	body := get()
	for _, want := range []string{"16 GiB of RAM", "gfx1201", "You can still start it"} {
		if !strings.Contains(body, want) {
			t.Errorf("needs box lacks %q:\n%s", want, body)
		}
	}
}

// A running fork instance shows on the GPU map as its whole GPU.
func TestForkOnGPUMap(t *testing.T) {
	s, port := forkInstanceServer(t)
	s.forks.AddInstance(forks.Instance{ID: "q", ForkID: "avalon", GPU: 1, Port: port})
	if got := s.forkReservationAllocations(0); len(got) != 0 {
		t.Errorf("stopped instance on the map: %+v", got)
	}
	if _, err := s.startForkInstance("q"); err != nil {
		t.Fatal(err)
	}
	got := s.forkReservationAllocations(3)
	if len(got) != 1 || got[0].GPUs[0] != 1 || got[0].TotalGB < 23 || !got[0].Reserved || !strings.Contains(got[0].ModelName, "whole GPU") {
		t.Errorf("map entry = %+v", got)
	}
}
