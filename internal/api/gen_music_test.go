package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/process"
)

// fakeAce is a stand-in ace-server with acestep.cpp's job API: POST /lm and
// /synth return a job ID, GET /job reports its status, and the synth result
// is multipart/mixed with an audio part and a latent part.
type fakeAce struct {
	srv     *httptest.Server
	mu      sync.Mutex
	nextID  int
	status  map[string]string
	kind    map[string]string
	lmReqs  []map[string]any
	cancels []string
	// hold keeps /lm jobs running until released; failLM fails them.
	hold   chan struct{}
	failLM bool
}

func newFakeAce(t *testing.T) *fakeAce {
	t.Helper()
	f := &fakeAce{status: map[string]string{}, kind: map[string]string{}}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeAce) serve(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	switch {
	case r.URL.Path == "/health":
		w.Write([]byte(`{"status":"ok"}`))
	case r.Method == "POST" && (r.URL.Path == "/lm" || r.URL.Path == "/synth"):
		f.nextID++
		id := fmt.Sprint(f.nextID)
		f.kind[id] = r.URL.Path
		f.status[id] = "done"
		if r.URL.Path == "/lm" {
			var req map[string]any
			json.NewDecoder(r.Body).Decode(&req)
			f.lmReqs = append(f.lmReqs, req)
			switch {
			case f.failLM:
				f.status[id] = "failed"
			case f.hold != nil:
				f.status[id] = "running"
				hold := f.hold
				go func() {
					<-hold
					f.mu.Lock()
					if f.status[id] == "running" {
						f.status[id] = "done"
					}
					f.mu.Unlock()
				}()
			}
		}
		fmt.Fprintf(w, `{"id":"%s"}`, id)
	case r.URL.Path == "/job" && r.URL.Query().Get("cancel") == "1":
		id := r.URL.Query().Get("id")
		f.cancels = append(f.cancels, id)
		f.status[id] = "cancelled"
	case r.URL.Path == "/job" && r.URL.Query().Get("result") == "1":
		id := r.URL.Query().Get("id")
		if f.kind[id] == "/lm" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`[{"caption":"bright folk","lyrics":"la la","bpm":96,"duration":30,"keyscale":"G major","timesignature":"4","seed":7}]`))
			return
		}
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		ap, _ := mw.CreatePart(textproto.MIMEHeader{"Content-Type": {"audio/mpeg"}})
		ap.Write([]byte("MP3DATA"))
		lp, _ := mw.CreatePart(textproto.MIMEHeader{"Content-Type": {"application/octet-stream"}})
		lp.Write([]byte("LATENT"))
		mw.Close()
		w.Header().Set("Content-Type", "multipart/mixed; boundary="+mw.Boundary())
		w.Write(buf.Bytes())
	case r.URL.Path == "/job":
		fmt.Fprintf(w, `{"status":"%s"}`, f.status[r.URL.Query().Get("id")])
	default:
		http.NotFound(w, r)
	}
}

// musicServer is vramServer with a complete ace set and a running ace
// instance on the fake's port.
func musicServer(t *testing.T) (*Server, *fakeAce) {
	t.Helper()
	s := vramServer(t)
	if err := s.gen.AddModelSet(generation.ModelSet{ID: "ace-xl", Name: "ACE XL", Kind: generation.KindAce, Complete: true,
		Components: []generation.Component{{Slot: "lm", Path: "/m/acestep-5Hz-lm-4B-Q8_0.gguf"}, {Slot: "dit", Path: "/m/acestep-v15-xl-sft-Q8_0.gguf"}}}); err != nil {
		t.Fatal(err)
	}
	f := newFakeAce(t)
	runFakeAce(t, s, "music", f)
	return s, f
}

func runFakeAce(t *testing.T, s *Server, id string, f *fakeAce) {
	t.Helper()
	addInstance(t, s, generation.Instance{ID: id, ModelSetID: "ace-xl", Port: (&fakeBackend{srv: f.srv}).port(t), Enabled: true, ReservedMiB: 8192})
	bin := filepath.Join(t.TempDir(), "fake")
	os.WriteFile(bin, []byte("#!/bin/sh\nexec sleep 30\n"), 0o755)
	if err := s.genMgr.Start(id, generation.KindAce, process.SupervisorConfig{
		Name: id, BinaryPath: bin, Env: os.Environ(), HealthURL: f.srv.URL + "/health", HealthInterval: 20 * time.Millisecond,
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

func postMusic(s *Server, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/v1/music/jobs", strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.handleMusicCreate(rec, r)
	return rec
}

func waitMusic(t *testing.T, s *Server, id string, status string) musicJob {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if j, _ := s.genMusic.view(id); j.Status == status {
			return j
		}
		time.Sleep(20 * time.Millisecond)
	}
	j, _ := s.genMusic.view(id)
	t.Fatalf("job %s is %q (%s), want %q", id, j.Status, j.Error, status)
	return j
}

func createdID(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var out struct{ ID string }
	json.Unmarshal(rec.Body.Bytes(), &out)
	if rec.Code != http.StatusAccepted || out.ID == "" {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	return out.ID
}

func TestMusicJobSucceeds(t *testing.T) {
	s, f := musicServer(t)
	id := createdID(t, postMusic(s, `{"model":"music","caption":"bright folk","lyrics":"la la","duration":30,"seed":7}`))
	j := waitMusic(t, s, id, musicSucceeded)
	if j.Result["bpm"] != float64(96) || j.Result["keyscale"] != "G major" {
		t.Errorf("result = %v", j.Result)
	}
	f.mu.Lock()
	req := f.lmReqs[0]
	f.mu.Unlock()
	if req["use_cot_caption"] != false || req["seed"] != float64(7) || req["duration"] != float64(30) {
		t.Errorf("ace /lm request = %v; want the caption kept as written", req)
	}

	rec := httptest.NewRecorder()
	s.handleMusicGet(rec, withID(httptest.NewRequest("GET", "/", nil), id))
	if !strings.Contains(rec.Body.String(), `"audio_url":"/v1/music/jobs/`+id+`/audio"`) {
		t.Errorf("record = %s", rec.Body.String())
	}
	rec = httptest.NewRecorder()
	s.handleMusicAudio(rec, withID(httptest.NewRequest("GET", "/", nil), id))
	if rec.Code != http.StatusOK || rec.Body.String() != "MP3DATA" || rec.Header().Get("Content-Type") != "audio/mpeg" {
		t.Errorf("audio: %d %q %q", rec.Code, rec.Body.String(), rec.Header().Get("Content-Type"))
	}
	if h := s.genQueue.History(0); len(h) != 1 || h[0].Kind != generation.KindAce {
		t.Errorf("queue history %+v", h)
	}
}

// A job's phase follows its steps and is empty once it ends.
func TestMusicJobPhases(t *testing.T) {
	s, _ := musicServer(t)
	phases := recordPhases(&s.genMusic, func(j musicJob) string { return j.Phase })
	busy, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	rec := postMusic(s, `{"caption":"x"}`)
	if !strings.Contains(rec.Body.String(), `"phase":"queued"`) {
		t.Errorf("create: %s", rec.Body.String())
	}
	id := createdID(t, rec)
	busy.Done()
	j := waitMusic(t, s, id, musicSucceeded)
	if want := []string{"composing", "synthesizing", "saving", ""}; !slices.Equal(phases(), want) {
		t.Errorf("phases = %v, want %v", phases(), want)
	}
	rec = httptest.NewRecorder()
	s.handleMusicGet(rec, withID(httptest.NewRequest("GET", "/", nil), id))
	if j.Phase != "" || strings.Contains(rec.Body.String(), `"phase"`) {
		t.Errorf("a finished job shows a phase: %s", rec.Body.String())
	}
}

// On a free GPU the job starts at once, and the first answer says so.
func TestMusicJobStartsAtOnceOnAFreeGPU(t *testing.T) {
	s, f := musicServer(t)
	f.mu.Lock()
	f.hold = make(chan struct{})
	f.mu.Unlock()
	defer close(f.hold)
	rec := postMusic(s, `{"caption":"x"}`)
	if !strings.Contains(rec.Body.String(), `"status":"running"`) || !strings.Contains(rec.Body.String(), `"phase":"composing"`) {
		t.Errorf("create: %d %s", rec.Code, rec.Body.String())
	}
	s.cancelMusic(createdID(t, rec))
}

func TestMusicCancelQueuedAndRunning(t *testing.T) {
	s, f := musicServer(t)

	// Queued: the GPU is busy with another job.
	busy, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	queued := createdID(t, postMusic(s, `{"caption":"x"}`))
	if j, _ := s.genMusic.view(queued); j.Status != musicQueued || j.Position != 1 {
		t.Errorf("queued job = %s position %d", j.Status, j.Position)
	}
	rec := httptest.NewRecorder()
	s.handleMusicDelete(rec, withID(httptest.NewRequest("DELETE", "/", nil), queued))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete: %d", rec.Code)
	}
	waitMusic(t, s, queued, musicCancelled)
	busy.Done()

	// Running: cancelled through ace-server.
	f.mu.Lock()
	f.hold = make(chan struct{})
	f.mu.Unlock()
	running := createdID(t, postMusic(s, `{"caption":"x"}`))
	waitMusic(t, s, running, musicRunning)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		f.mu.Lock()
		n := len(f.lmReqs)
		f.mu.Unlock()
		if n >= 1 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	time.Sleep(100 * time.Millisecond)
	s.cancelMusic(running)
	waitMusic(t, s, running, musicCancelled)
	time.Sleep(aceJobPoll * 3)
	f.mu.Lock()
	cancels := len(f.cancels)
	f.mu.Unlock()
	if cancels != 1 {
		t.Errorf("ace-server got %d cancels, want 1", cancels)
	}
	if s.genQueue.Load(0) != 0 {
		t.Error("the GPU is still booked after the cancelled job")
	}
	close(f.hold)
}

func TestMusicBackendFailure(t *testing.T) {
	s, f := musicServer(t)
	f.failLM = true
	id := createdID(t, postMusic(s, `{"caption":"x"}`))
	j := waitMusic(t, s, id, musicFailed)
	if !strings.Contains(j.Error, "composing") || !strings.Contains(j.Error, "failed") {
		t.Errorf("error = %q", j.Error)
	}
}

func TestMusicRefusals(t *testing.T) {
	s, _ := musicServer(t)
	for body, want := range map[string]string{
		`{"model":"nope","caption":"x"}`:         "model_not_found",
		`{"caption":""}`:                         "caption is required",
		`{"caption":"x","output_format":"flac"}`: "output_format",
		`not json`:                               "not valid JSON",
	} {
		rec := postMusic(s, body)
		if !strings.Contains(rec.Body.String(), want) {
			t.Errorf("%s: %d %s, want %q", body, rec.Code, rec.Body.String(), want)
		}
	}
	rec := httptest.NewRecorder()
	s.handleMusicAudio(rec, withID(httptest.NewRequest("GET", "/", nil), "0000000000000000"))
	if rec.Code != http.StatusNotFound {
		t.Errorf("audio of an unknown job: %d", rec.Code)
	}
	s.cfg.GenerationEnabled = false
	if rec := postMusic(s, `{"caption":"x"}`); rec.Code != http.StatusNotFound {
		t.Errorf("with generation off: %d", rec.Code)
	}
}

// The official ACE-Step server's flow: release_task, query_result, then
// the audio from the file path in the result.
func TestAceStepCompatibleFlow(t *testing.T) {
	s, _ := musicServer(t)
	post := func(h http.HandlerFunc, body string) map[string]any {
		r := httptest.NewRequest("POST", "/", strings.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		h(rec, r)
		var out map[string]any
		json.Unmarshal(rec.Body.Bytes(), &out)
		out["_code"] = rec.Code
		return out
	}

	// An official model name falls back to the only music instance.
	out := post(s.handleReleaseTask, `{"prompt":"bright folk","lyrics":"la la","audio_duration":30,"model":"acestep-v15-turbo","batch_size":1}`)
	if out["code"] != float64(200) || out["error"] != nil {
		t.Fatalf("release_task: %v", out)
	}
	task := out["data"].(map[string]any)["task_id"].(string)
	waitMusic(t, s, task, musicSucceeded)

	out = post(s.handleQueryResult, `{"task_id_list":["`+task+`"]}`)
	entry := out["data"].([]any)[0].(map[string]any)
	if entry["status"] != float64(1) {
		t.Fatalf("query_result status = %v", entry["status"])
	}
	var items []map[string]any
	json.Unmarshal([]byte(entry["result"].(string)), &items)
	file := items[0]["file"].(string)
	if !strings.HasPrefix(file, "/v1/audio?path="+task+".mp3") || items[0]["dit_model"] != "acestep-v15-xl-sft-Q8_0" ||
		items[0]["metas"].(map[string]any)["keyscale"] != "G major" {
		t.Errorf("result item = %v", items[0])
	}

	rec := httptest.NewRecorder()
	s.handleAceStepAudio(rec, httptest.NewRequest("GET", file, nil))
	if rec.Code != http.StatusOK || rec.Body.String() != "MP3DATA" {
		t.Errorf("/v1/audio: %d %q", rec.Code, rec.Body.String())
	}
	for _, bad := range []string{"../../config/llama-toolchest.yaml", "x.mp3", "%2Fetc%2Fpasswd"} {
		rec := httptest.NewRecorder()
		s.handleAceStepAudio(rec, httptest.NewRequest("GET", "/v1/audio?path="+bad, nil))
		if rec.Code != http.StatusBadRequest {
			t.Errorf("path %q: %d, want 400", bad, rec.Code)
		}
	}

	if out := post(s.handleReleaseTask, `{"prompt":"x","batch_size":2}`); out["_code"] != http.StatusBadRequest || out["detail"] != "batch_size > 1 is not supported" {
		t.Errorf("batch_size 2: %v", out)
	}
	if out := post(s.handleQueryResult, `{"task_id_list":["nope"]}`); out["data"].([]any)[0].(map[string]any)["status"] != float64(2) {
		t.Errorf("unknown task: %v", out)
	}

	// With two music instances an unknown name cannot fall back.
	runFakeAce(t, s, "music-2", newFakeAce(t))
	if out := post(s.handleReleaseTask, `{"prompt":"x","model":"acestep-v15-turbo"}`); out["_code"] != http.StatusNotFound ||
		!strings.Contains(out["detail"].(string), "music, music-2") {
		t.Errorf("unknown model with two instances: %v", out)
	}
}

// The root ACE-Step routes need the API key like /v1 does.
func TestAceStepRoutesNeedTheAPIKey(t *testing.T) {
	s, _ := musicServer(t)
	s.cfg.APIKey = "secret"
	router := s.buildRouter()
	for _, path := range []string{"/release_task", "/query_result"} {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest("POST", path, strings.NewReader(`{}`)))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s without a key: %d, want 401", path, rec.Code)
		}
	}
}

func TestMusicJobsPrune(t *testing.T) {
	s, _ := musicServer(t)
	dir := t.TempDir()
	old, recent := time.Now().Add(-25*time.Hour), time.Now()
	for i, fin := range []*time.Time{&old, &recent, nil} {
		id := fmt.Sprintf("%016x", i)
		file := filepath.Join(dir, id+".mp3")
		os.WriteFile(file, []byte("x"), 0o644)
		s.genMusic.add(&musicJob{ID: id, Status: musicSucceeded, Finished: fin, audioPath: file})
	}
	s.genMusic.prune(time.Now())
	if _, ok := s.genMusic.view("0000000000000000"); ok {
		t.Error("a job finished 25 hours ago was kept")
	}
	if _, err := os.Stat(filepath.Join(dir, "0000000000000000.mp3")); !os.IsNotExist(err) {
		t.Error("its audio file was kept")
	}
	for _, id := range []string{"0000000000000001", "0000000000000002"} {
		if _, ok := s.genMusic.view(id); !ok {
			t.Errorf("job %s was pruned", id)
		}
	}
}
