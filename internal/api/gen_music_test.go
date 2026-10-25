package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	// synthReqs and synthFiles are each /synth request and its file parts
	// (field: content).
	synthReqs  []map[string]any
	synthFiles []map[string]string
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
		if r.URL.Path == "/synth" {
			f.recordSynth(r)
		}
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

func (f *fakeAce) recordSynth(r *http.Request) {
	var req map[string]any
	files := map[string]string{}
	if r.ParseMultipartForm(1<<20) == nil {
		json.Unmarshal([]byte(r.FormValue("request")), &req)
		for field, fhs := range r.MultipartForm.File {
			rd, _ := fhs[0].Open()
			b, _ := io.ReadAll(rd)
			files[field] = string(b)
		}
	} else {
		json.NewDecoder(r.Body).Decode(&req)
	}
	f.synthReqs = append(f.synthReqs, req)
	f.synthFiles = append(f.synthFiles, files)
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

// postMusicMultipart posts a multipart music job: the JSON in a "request"
// part and each file in a part of its own.
func postMusicMultipart(s *Server, request string, files map[string]string) *httptest.ResponseRecorder {
	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("request", request)
	for field, content := range files {
		part, _ := mw.CreateFormFile(field, field+".mp3")
		part.Write([]byte(content))
	}
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/music/jobs", &buf)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.handleMusicCreate(rec, r)
	return rec
}

// A cover skips /lm and sends ace-server the task, its settings and both
// songs; the job keeps no copy of them once it is done.
func TestMusicCoverFromUpload(t *testing.T) {
	s, f := musicServer(t)
	phases := recordPhases(&s.genMusic, func(j musicJob) string { return j.Phase })
	rec := postMusicMultipart(s, `{"model":"music","caption":"smoky jazz","lyrics":"new words","task_type":"cover","audio_cover_strength":0.3}`,
		map[string]string{"audio": "SRC", "ref_audio": "REF"})
	j := waitMusic(t, s, createdID(t, rec), musicSucceeded)
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.lmReqs) != 0 {
		t.Errorf("a cover ran /lm: %v", f.lmReqs)
	}
	if len(f.synthReqs) != 1 {
		t.Fatalf("/synth requests = %d", len(f.synthReqs))
	}
	req, files := f.synthReqs[0], f.synthFiles[0]
	if req["task_type"] != "cover" || req["audio_cover_strength"] != 0.3 || req["caption"] != "smoky jazz" || req["lyrics"] != "new words" {
		t.Errorf("/synth request = %v", req)
	}
	if files["audio"] != "SRC" || files["ref_audio"] != "REF" {
		t.Errorf("/synth files = %v", files)
	}
	if !j.Request.SourceAudio || !j.Request.ReferenceAudio || j.Request.source != nil || j.Request.ref != nil {
		t.Errorf("finished job's request = %+v", j.Request)
	}
	if j.Result["caption"] != "smoky jazz" {
		t.Errorf("result = %v", j.Result)
	}
	if want := []string{"synthesizing", "saving", ""}; !slices.Equal(phases(), want) {
		t.Errorf("phases = %v, want %v", phases(), want)
	}
}

// Text to music with a timbre reference composes first, then hands the
// reference to /synth.
func TestMusicText2MusicWithReference(t *testing.T) {
	s, f := musicServer(t)
	rec := postMusicMultipart(s, `{"caption":"bright folk"}`, map[string]string{"ref_audio": "REF"})
	waitMusic(t, s, createdID(t, rec), musicSucceeded)
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.lmReqs) != 1 || f.synthFiles[0]["ref_audio"] != "REF" || f.synthFiles[0]["audio"] != "" {
		t.Errorf("lm %d, synth files %v", len(f.lmReqs), f.synthFiles)
	}
	if _, ok := f.synthReqs[0]["task_type"]; ok {
		t.Errorf("text2music sent a task_type: %v", f.synthReqs[0])
	}
}

func TestMusicTaskRefusals(t *testing.T) {
	s, _ := musicServer(t)
	turbo := generation.ModelSet{ID: "ace-turbo", Kind: generation.KindAce,
		Components: []generation.Component{{Slot: "dit", Path: "/m/acestep-v15-turbo-Q8_0.gguf"}}}
	src := &aceFile{field: "audio", data: []byte("SRC")}
	if code, msg := checkMusicRequest(turbo, musicRequest{Caption: "x", TaskType: "lego", Track: "drums", source: src}, "caption"); code != "task_not_supported" {
		t.Errorf("lego on turbo: %s %s", code, msg)
	}

	audio := map[string]string{"audio": "SRC"}
	for _, c := range []struct {
		request string
		files   map[string]string
		want    string
	}{
		{`{"model":"music","caption":"x","task_type":"remix"}`, audio, "task_type must be one of"},
		{`{"model":"music","caption":"x","task_type":"cover"}`, nil, "cover needs source audio"},
		{`{"model":"music","caption":"x"}`, audio, "only used by the other tasks"},
		{`{"model":"music","caption":"","task_type":"cover"}`, audio, "caption is required"},
		{`{"model":"music","task_type":"lego","track":"kazoo","caption":"x"}`, audio, "unknown track"},
		{`{"model":"music","task_type":"cover","track":"drums","caption":"x"}`, audio, "track is only used"},
		{`{"model":"music","task_type":"cover","audio_cover_strength":1.5,"caption":"x"}`, audio, "between 0 and 1"},
		{`{"model":"music","task_type":"repaint","audio_cover_strength":0.5,"caption":"x"}`, audio, "only used by cover"},
		{`{"model":"music","task_type":"repaint","repainting_start":20,"repainting_end":10,"caption":"x"}`, audio, "must be after"},
		{`{"model":"music","task_type":"cover","repainting_start":5,"caption":"x"}`, audio, "only used by repaint and lego"},
		{`{"model":"music","caption":"x"`, audio, "not valid JSON"},
	} {
		rec := postMusicMultipart(s, c.request, c.files)
		if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%s: %d %s, want %q", c.request, rec.Code, rec.Body.String(), c.want)
		}
	}
	// Extract needs no caption; a complete takes several tracks; repaint
	// to the end of the source is -1.
	for _, request := range []string{
		`{"model":"music","task_type":"extract","track":"vocals"}`,
		`{"model":"music","task_type":"complete","track":"drums | bass","caption":"x"}`,
		`{"model":"music","task_type":"repaint","repainting_start":-10,"repainting_end":-1,"caption":"x"}`,
	} {
		if rec := postMusicMultipart(s, request, audio); rec.Code != http.StatusAccepted {
			t.Errorf("%s: %d %s", request, rec.Code, rec.Body.String())
		}
	}
}

func TestAceTasks(t *testing.T) {
	set := func(dit string) generation.ModelSet {
		return generation.ModelSet{Kind: generation.KindAce, Components: []generation.Component{{Slot: "dit", File: dit}}}
	}
	if got := aceTasks(set("acestep-v15-turbo-Q8_0.gguf")); slices.Contains(got, taskLego) || !slices.Contains(got, taskRepaint) {
		t.Errorf("turbo tasks = %v", got)
	}
	if got := aceTasks(set("acestep-v15-xl-sft-Q8_0.gguf")); !slices.Equal(got, musicTasks) {
		t.Errorf("sft tasks = %v", got)
	}
	if !slices.Contains(musicTasks, taskLego) {
		t.Error("aceTasks changed musicTasks")
	}
}

// The official server's multipart form: each field in a part of its own and
// the song in src_audio.
func TestAceStepReleaseTaskMultipart(t *testing.T) {
	s, f := musicServer(t)
	post := func(fields map[string]string, files map[string]string) *httptest.ResponseRecorder {
		var buf bytes.Buffer
		mw := multipart.NewWriter(&buf)
		for k, v := range fields {
			mw.WriteField(k, v)
		}
		for k, v := range files {
			part, _ := mw.CreateFormFile(k, k+".wav")
			part.Write([]byte(v))
		}
		mw.Close()
		r := httptest.NewRequest("POST", "/release_task", &buf)
		r.Header.Set("Content-Type", mw.FormDataContentType())
		rec := httptest.NewRecorder()
		s.handleReleaseTask(rec, r)
		return rec
	}
	rec := post(map[string]string{"prompt": "x", "task_type": "repaint", "repainting_start": "10", "repainting_end": "20", "thinking": "true"},
		map[string]string{"src_audio": "SRC"})
	var out struct {
		Data struct {
			TaskID string `json:"task_id"`
		}
	}
	json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Data.TaskID == "" {
		t.Fatalf("release_task: %d %s", rec.Code, rec.Body.String())
	}
	waitMusic(t, s, out.Data.TaskID, musicSucceeded)
	f.mu.Lock()
	req, files := f.synthReqs[0], f.synthFiles[0]
	f.mu.Unlock()
	if req["task_type"] != "repaint" || req["repainting_start"] != float64(10) || req["repainting_end"] != float64(20) || files["audio"] != "SRC" {
		t.Errorf("/synth got %v %v", req, files)
	}

	for _, c := range []struct {
		fields map[string]string
		want   string
	}{
		{map[string]string{"prompt": "x", "src_audio_path": "/srv/song.mp3", "task_type": "cover"}, "upload the files"},
		{map[string]string{"prompt": "x", "repainting_start": "soon"}, "not a valid value"},
	} {
		if rec := post(c.fields, nil); rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), c.want) {
			t.Errorf("%v: %d %s, want %q", c.fields, rec.Code, rec.Body.String(), c.want)
		}
	}
}
