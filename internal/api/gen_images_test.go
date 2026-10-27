package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// fakeBackend is a stand-in sd-server: it answers health checks and image
// requests, recording what it got and how many requests ran at once.
type fakeBackend struct {
	srv          *httptest.Server
	mu           sync.Mutex
	bodies       []string
	active, peak atomic.Int32
	delay        time.Duration
}

func newFakeBackend(t *testing.T, delay time.Duration) *fakeBackend {
	t.Helper()
	fb := &fakeBackend{delay: delay}
	fb.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			return
		}
		n := fb.active.Add(1)
		for {
			p := fb.peak.Load()
			if n <= p || fb.peak.CompareAndSwap(p, n) {
				break
			}
		}
		body, _ := io.ReadAll(r.Body)
		fb.mu.Lock()
		fb.bodies = append(fb.bodies, r.URL.Path+" "+string(body))
		fb.mu.Unlock()
		time.Sleep(fb.delay)
		fb.active.Add(-1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"b64_json":"aW1n"}]}`))
	}))
	t.Cleanup(fb.srv.Close)
	return fb
}

func (fb *fakeBackend) port(t *testing.T) int {
	_, p, _ := net.SplitHostPort(fb.srv.Listener.Addr().String())
	n, _ := strconv.Atoi(p)
	return n
}

// runFakeInstance stores an enabled, reserved instance on the fake
// backend's port and makes the manager report it running.
func runFakeInstance(t *testing.T, s *Server, id, set string, gpu int, fb *fakeBackend) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script fake")
	}
	addInstance(t, s, generation.Instance{ID: id, ModelSetID: set, GPU: gpu, Port: fb.port(t), Enabled: true, ReservedMiB: 4096})
	bin := filepath.Join(t.TempDir(), "fake")
	os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755)
	if err := s.genMgr.Start(id, generation.KindSD, process.SupervisorConfig{
		Name: id, BinaryPath: bin, Env: os.Environ(), HealthURL: fb.srv.URL + "/v1/models", HealthInterval: 20 * time.Millisecond,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.genMgr.Stop(id) })
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := s.genMgr.WaitRunning(ctx, id); err != nil {
		t.Fatal(err)
	}
}

func imageReq(body string) *http.Request {
	r := httptest.NewRequest("POST", "/v1/images/generations", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	return r
}

func errCode(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var e struct {
		Error struct{ Code, Message string }
	}
	json.Unmarshal(rec.Body.Bytes(), &e)
	return e.Error.Code + ": " + e.Error.Message
}

func TestImageRoutedByInstanceID(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)

	rec := httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"ming","prompt":"a fox","size":"1024x1024"}`))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "aW1n") {
		t.Fatalf("response %d %s", rec.Code, rec.Body.String())
	}
	if len(fb.bodies) != 1 || !strings.Contains(fb.bodies[0], `"prompt":"a fox"`) || !strings.HasPrefix(fb.bodies[0], "/v1/images/generations ") {
		t.Errorf("backend got %q", fb.bodies)
	}
	if h := s.genQueue.History(0); len(h) != 1 || h[0].InstanceID != "ming" {
		t.Errorf("queue history %+v", h)
	}
}

// With model naming a set served by two instances, the one whose GPU has
// fewer jobs takes it.
func TestImageRoutedBySetToLeastBusyGPU(t *testing.T) {
	s := vramServer(t)
	a, b := newFakeBackend(t, 0), newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming-a", "ming-image", 0, a)
	runFakeInstance(t, s, "ming-b", "ming-image", 1, b)
	busy, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	defer busy.Done()

	rec := httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"ming-image","prompt":"x"}`))
	if rec.Code != http.StatusOK || len(b.bodies) != 1 || len(a.bodies) != 0 {
		t.Errorf("code %d; GPU 0 got %d, GPU 1 got %d, want GPU 1", rec.Code, len(a.bodies), len(b.bodies))
	}

	// With model empty and two instances, the client has to choose.
	rec = httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"prompt":"x"}`))
	if rec.Code != http.StatusBadRequest || !strings.Contains(errCode(t, rec), "ming-a, ming-b") {
		t.Errorf("empty model with two backends: %d %s", rec.Code, errCode(t, rec))
	}
}

func TestImageRefusals(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "stopped", Port: 8090, Enabled: true, ReservedMiB: 4096})
	addInstance(t, s, generation.Instance{ID: "unmeasured", Port: 8091, Enabled: true})

	for model, want := range map[string]string{
		"nope":       "model_not_found",
		"stopped":    "backend stopped is not running",
		"unmeasured": "backend unmeasured has no VRAM reservation yet",
	} {
		rec := httptest.NewRecorder()
		s.handleGenImages(rec, imageReq(`{"model":"`+model+`","prompt":"x"}`))
		if got := errCode(t, rec); !strings.Contains(got, want) {
			t.Errorf("%s: %d %s, want %q", model, rec.Code, got, want)
		}
	}
	s.genMgr.SetMeasuring("stopped", true)
	rec := httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"stopped","prompt":"x"}`))
	if rec.Code != http.StatusServiceUnavailable || !strings.Contains(errCode(t, rec), "being measured") {
		t.Errorf("measuring: %d %s", rec.Code, errCode(t, rec))
	}

	s.cfg.GenerationEnabled = false
	rec = httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"x"}`))
	if rec.Code != http.StatusNotFound {
		t.Errorf("with generation off: %d", rec.Code)
	}
}

// An edit to a set without the edit capability is refused; the model is
// read from the multipart form.
func TestImageEditsNeedTheCapability(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("model", "ming")
	mw.WriteField("prompt", "make it blue")
	fw, _ := mw.CreateFormFile("image", "in.png")
	fw.Write([]byte("PNGDATA"))
	mw.Close()
	edit := func() *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "/v1/images/edits", bytes.NewReader(buf.Bytes()))
		r.Header.Set("Content-Type", mw.FormDataContentType())
		rec := httptest.NewRecorder()
		s.handleGenImages(rec, r)
		return rec
	}
	if rec := edit(); rec.Code != http.StatusBadRequest || !strings.Contains(errCode(t, rec), "does not support image edits") {
		t.Errorf("edit without the capability: %d %s", rec.Code, errCode(t, rec))
	}

	set, _ := s.gen.ModelSet("ming-image")
	set.Capabilities = []string{"edit"}
	s.gen.UpdateModelSet(set)
	if rec := edit(); rec.Code != http.StatusOK {
		t.Fatalf("edit with the capability: %d %s", rec.Code, rec.Body.String())
	}
	if len(fb.bodies) != 1 || !strings.Contains(fb.bodies[0], "PNGDATA") || !strings.HasPrefix(fb.bodies[0], "/v1/images/edits ") {
		t.Errorf("the multipart body was not forwarded unchanged: %.80q", fb.bodies)
	}
}

// Requests for one GPU run one at a time, even when they arrive together.
func TestImageRequestsOnOneGPUTakeTurns(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 100*time.Millisecond)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rec := httptest.NewRecorder()
			s.handleGenImages(rec, imageReq(`{"model":"ming","prompt":"x"}`))
			if rec.Code != http.StatusOK {
				t.Errorf("request: %d", rec.Code)
			}
		}()
	}
	wg.Wait()
	if p := fb.peak.Load(); p != 1 {
		t.Errorf("%d requests ran at once on one GPU, want 1", p)
	}
	h := s.genQueue.History(0)
	if len(h) != 3 {
		t.Fatalf("history %d jobs", len(h))
	}
	for i := 1; i < len(h); i++ {
		if h[i].Started.Before(h[i-1].Finished) {
			t.Error("jobs overlapped")
		}
	}
}

func TestImageQueueFull(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	var held []*generation.Ticket
	for i := 0; i <= generation.MaxWaitingPerGPU; i++ {
		tk, _ := s.genQueue.Submit(0, "other", generation.KindSD)
		held = append(held, tk)
	}
	defer func() {
		for _, tk := range held {
			tk.Done()
		}
	}()
	rec := httptest.NewRecorder()
	s.handleGenImages(rec, imageReq(`{"model":"ming","prompt":"x"}`))
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") != "30" {
		t.Errorf("full queue: %d Retry-After=%q", rec.Code, rec.Header().Get("Retry-After"))
	}
}

func TestV1ModelsListsInstances(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 4096})
	addInstance(t, s, generation.Instance{ID: "off", Port: 8082, Enabled: false})
	list := func() string {
		rec := httptest.NewRecorder()
		s.handleV1Models(rec, httptest.NewRequest("GET", "/v1/models", nil))
		return rec.Body.String()
	}
	on := list()
	if !strings.Contains(on, `"id":"ming"`) || !strings.Contains(on, `"kind":"image"`) || strings.Contains(on, `"id":"off"`) {
		t.Errorf("list with generation on: %s", on)
	}
	rec := httptest.NewRecorder()
	r := httptest.NewRequest("GET", "/v1/models/ming", nil)
	s.handleV1Model(rec, withModelParam(r, "ming"))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"model_set":"ming-image"`) {
		t.Errorf("single lookup: %d %s", rec.Code, rec.Body.String())
	}

	s.cfg.GenerationEnabled = false
	if off := list(); strings.Contains(off, "kind") || strings.Contains(off, "ming") {
		t.Errorf("list with generation off changed: %s", off)
	}
}

// meta.api lists the route groups for each generation entry; text
// entries have none. Music entries list their tasks too.
func TestV1ModelsMetaAPI(t *testing.T) {
	s, _ := musicServer(t)
	addInstance(t, s, generation.Instance{ID: "ming", Port: 8081, Enabled: true, ReservedMiB: 4096})
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "klein", Name: "klein", Kind: generation.KindSD, Capabilities: []string{"edit"}}); err != nil {
		t.Fatal(err)
	}
	addInstance(t, s, generation.Instance{ID: "klein", ModelSetID: "klein", Port: 8082, Enabled: true, ReservedMiB: 4096})
	if err := s.registry.Add(&models.Model{ID: "chat", ModelID: "org/chat", Filename: "chat.gguf"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	s.handleV1Models(rec, httptest.NewRequest("GET", "/v1/models", nil))
	var out struct {
		Data []struct {
			ID   string
			Meta map[string]any
		}
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	want := map[string]string{
		"ming":  "[images.sync images.jobs]",
		"klein": "[images.sync images.jobs images.edits]",
		"music": "[music.jobs music.ace_step]",
	}
	for _, m := range out.Data {
		api, has := m.Meta["api"]
		if tasks, ok := m.Meta["tasks"]; (m.ID == "music") != ok || ok && fmt.Sprint(tasks) != fmt.Sprint(musicTasks) {
			t.Errorf("%s: meta.tasks = %v", m.ID, tasks)
		}
		if w, ok := want[m.ID]; ok {
			if fmt.Sprint(api) != w {
				t.Errorf("%s: meta.api = %v, want %s", m.ID, api, w)
			}
			delete(want, m.ID)
		} else if has {
			t.Errorf("text entry %s has meta.api %v", m.ID, api)
		}
	}
	if len(want) != 0 || len(out.Data) != 4 {
		t.Errorf("missing entries: %v (list %s)", want, rec.Body.String())
	}
}

func TestMeasureRefusedWhileQueueBusy(t *testing.T) {
	s := vramServer(t)
	addInstance(t, s, generation.Instance{ID: "m", Port: 8081, Enabled: true})
	inst, _ := s.gen.Instance("m")
	tk, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	if err := s.canMeasure(inst); err == nil || !strings.Contains(err.Error(), "wait for its image and music jobs") {
		t.Errorf("measure with a job running: %v", err)
	}
	tk.Done()
	if err := s.canMeasure(inst); err != nil {
		t.Errorf("idle: %v", err)
	}
}

// The image routes sit before the LLM proxy's catch-all.
func TestImageRouteRegistered(t *testing.T) {
	s := vramServer(t)
	s.cfg.GenerationEnabled = false
	rec := httptest.NewRecorder()
	s.buildRouter().ServeHTTP(rec, imageReq(`{"model":"x"}`))
	if !strings.Contains(rec.Body.String(), "image generation is not enabled") {
		t.Errorf("POST /v1/images/generations did not reach the image handler: %d %s", rec.Code, rec.Body.String())
	}
}

// withModelParam sets the {model} URL parameter of a request.
func withModelParam(r *http.Request, model string) *http.Request {
	rc := chi.NewRouteContext()
	rc.URLParams.Add("model", model)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc))
}

// With nothing enabled, /v1/models lists data as [], not null, which
// some OpenAI clients cannot read.
func TestV1ModelsEmptyListIsArray(t *testing.T) {
	s := newTestServer(t)
	rec := httptest.NewRecorder()
	s.handleV1Models(rec, httptest.NewRequest("GET", "/v1/models", nil))
	if !strings.Contains(rec.Body.String(), `"data":[]`) {
		t.Errorf("body = %s, want data as []", rec.Body)
	}
}
