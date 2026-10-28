package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/generation"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

func postImageJob(s *Server, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/v1/images/jobs", strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleImageJobCreate(rec, r)
	return rec
}

func waitImageJob(t *testing.T, s *Server, id, status string) imageJob {
	t.Helper()
	var j imageJob
	testutil.EventuallyMsg(t, func() bool {
		j, _ = s.genImageJobs.view(id)
		return j.Status == status
	}, func() string { return fmt.Sprintf("job %s is %q (%s), want %q", id, j.Status, j.Error, status) })
	return j
}

// getImage requests /v1/images/jobs/{id}/image/{n}.
func getImage(s *Server, id, n string) *httptest.ResponseRecorder {
	rc := chi.NewRouteContext()
	rc.URLParams.Add("id", id)
	rc.URLParams.Add("n", n)
	r := httptest.NewRequest("GET", "/", nil)
	rec := httptest.NewRecorder()
	s.handleImageJobImage(rec, r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rc)))
	return rec
}

// recordPhases records every status and phase a store's jobs go through,
// leaving out repeats.
func recordPhases[T jobRecord[T]](store *jobStore[T], label func(T) string) func() []string {
	var mu sync.Mutex
	var seen []string
	store.onUpdate = func(j T) {
		mu.Lock()
		defer mu.Unlock()
		if l := label(j); len(seen) == 0 || seen[len(seen)-1] != l {
			seen = append(seen, l)
		}
	}
	return func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(seen)
	}
}

func TestImageJobSucceeds(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	phases := recordPhases(&s.genImageJobs, func(j imageJob) string { return j.Status + "/" + j.Phase })

	// The GPU is busy, so the job waits.
	busy, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	rec := postImageJob(s, `{"model":"ming","prompt":"a red fox","size":"512x512"}`)
	var created map[string]any
	json.Unmarshal(rec.Body.Bytes(), &created)
	if rec.Code != http.StatusAccepted || created["status"] != "queued" || created["phase"] != "queued" || created["queue_position"] != float64(1) {
		t.Fatalf("create: %d %s", rec.Code, rec.Body.String())
	}
	id := created["id"].(string)
	if rec := getImage(s, id, "0"); rec.Code != http.StatusConflict || !strings.Contains(errCode(t, rec), "not_ready") {
		t.Errorf("image before success: %d %s", rec.Code, rec.Body.String())
	}
	busy.Done()

	j := waitImageJob(t, s, id, jobSucceeded)
	if want := []string{"running/generating", "running/saving", "succeeded/"}; !slices.Equal(phases(), want) {
		t.Errorf("phases = %v, want %v", phases(), want)
	}
	if len(j.Images) != 1 || j.Images[0].URL != "/v1/images/jobs/"+id+"/image/0" || j.Phase != "" {
		t.Errorf("job = %+v", j)
	}

	rec = httptest.NewRecorder()
	s.handleImageJobGet(rec, withID(httptest.NewRequest("GET", "/", nil), id))
	if body := rec.Body.String(); strings.Contains(body, `"phase"`) || !strings.Contains(body, `"request":{"model":"ming","prompt":"a red fox","size":"512x512"}`) {
		t.Errorf("record = %s", body)
	}
	if rec := getImage(s, id, "0"); rec.Code != http.StatusOK || rec.Body.String() != "img" || rec.Header().Get("Content-Type") != "image/png" {
		t.Errorf("image 0: %d %q %q", rec.Code, rec.Body.String(), rec.Header().Get("Content-Type"))
	}
	if rec := getImage(s, id, "1"); rec.Code != http.StatusNotFound {
		t.Errorf("image 1: %d", rec.Code)
	}
	if _, err := os.Stat(filepath.Join(s.cfg.DataDir, "gen-output", "images", id+"-0.png")); err != nil {
		t.Errorf("image file: %v", err)
	}
	if h := s.genQueue.History(0); len(h) != 2 || h[1].InstanceID != "ming" {
		t.Errorf("queue history %+v", h)
	}
}

// On a free GPU the job starts at once, and the first answer says so.
func TestImageJobStartsAtOnceOnAFreeGPU(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 200*time.Millisecond)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	rec := postImageJob(s, `{"model":"ming","prompt":"x"}`)
	var created map[string]any
	json.Unmarshal(rec.Body.Bytes(), &created)
	if rec.Code != http.StatusAccepted || created["status"] != "running" || created["phase"] != "generating" || created["queue_position"] != float64(0) {
		t.Errorf("create: %d %s", rec.Code, rec.Body.String())
	}
	if j := waitImageJob(t, s, created["id"].(string), jobSucceeded); j.Started == nil || j.Started.Before(j.Submitted) {
		t.Errorf("started = %v, submitted = %v", j.Started, j.Submitted)
	}
}

// The job shows back only model, prompt, size and n; the backend still
// gets the whole request.
func TestImageJobEchoLeavesOutLargeFields(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	body := `{"model":"ming","n":1,"init_image":"iVBORw0KGgoAAAA","prompt":"<sd_cpp_extra_args>{\"init_image\": \"iVBORw0KGgoBBBB\", \"strength\": 0.9}</sd_cpp_extra_args>RGBA, 4-channel, transparent background. a fox"}`
	id := createdID(t, postImageJob(s, body))
	waitImageJob(t, s, id, jobSucceeded)

	rec := httptest.NewRecorder()
	s.handleImageJobGet(rec, withID(httptest.NewRequest("GET", "/", nil), id))
	var out struct{ Request map[string]any }
	json.Unmarshal(rec.Body.Bytes(), &out)
	want := map[string]any{"model": "ming", "n": float64(1), "prompt": "RGBA, 4-channel, transparent background. a fox"}
	if fmt.Sprint(out.Request) != fmt.Sprint(want) {
		t.Errorf("request = %v, want %v", out.Request, want)
	}
	fb.mu.Lock()
	defer fb.mu.Unlock()
	if len(fb.bodies) != 1 || fb.bodies[0] != "/v1/images/generations "+body {
		t.Errorf("backend got %.120q", fb.bodies)
	}
}

func TestImageJobQueueFull(t *testing.T) {
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
	rec := postImageJob(s, `{"model":"ming","prompt":"x"}`)
	if rec.Code != http.StatusTooManyRequests || rec.Header().Get("Retry-After") != "30" || !strings.Contains(errCode(t, rec), "queue_full") {
		t.Errorf("full queue: %d Retry-After=%q %s", rec.Code, rec.Header().Get("Retry-After"), rec.Body.String())
	}
}

func TestImageJobCancelQueuedAndRunning(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 300*time.Millisecond)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	del := func(id string) {
		t.Helper()
		rec := httptest.NewRecorder()
		s.handleImageJobDelete(rec, withID(httptest.NewRequest("DELETE", "/", nil), id))
		if rec.Code != http.StatusNoContent {
			t.Fatalf("delete: %d", rec.Code)
		}
	}

	// Queued: it leaves the queue.
	busy, _ := s.genQueue.Submit(0, "other", generation.KindSD)
	queued := createdID(t, postImageJob(s, `{"model":"ming","prompt":"x"}`))
	if n := len(s.genQueue.Snapshot(0).Waiting); n != 1 {
		t.Fatalf("%d jobs waiting, want 1", n)
	}
	del(queued)
	waitImageJob(t, s, queued, jobCancelled)
	testutil.Eventually(t, func() bool { return len(s.genQueue.Snapshot(0).Waiting) == 0 },
		"jobs still waiting after the cancel")
	busy.Done()

	// Running: marked cancelled at once; sd-server's late answer is
	// discarded and the GPU is freed when it comes.
	running := createdID(t, postImageJob(s, `{"model":"ming","prompt":"x"}`))
	waitImageJob(t, s, running, jobRunning)
	del(running)
	if j, _ := s.genImageJobs.view(running); j.Status != jobCancelled || j.Phase != "" {
		t.Errorf("after the cancel: %s/%s", j.Status, j.Phase)
	}
	testutil.Eventually(t, func() bool { return s.genQueue.Load(0) == 0 },
		"the GPU is still booked after the cancelled job")
	// runImageJob drops the late result before it frees the GPU (the
	// ticket's Done is deferred), so with the GPU free the job is final.
	if j, _ := s.genImageJobs.view(running); j.Status != jobCancelled || len(j.Images) != 0 || len(j.paths) != 0 {
		t.Errorf("the late result appeared: %+v", j)
	}
	if files, _ := filepath.Glob(filepath.Join(s.imageDir(), running+"-*")); len(files) != 0 {
		t.Errorf("files of the cancelled job: %v", files)
	}
	if rec := getImage(s, running, "0"); rec.Code != http.StatusConflict {
		t.Errorf("image of the cancelled job: %d", rec.Code)
	}
}

func TestImageJobRefusals(t *testing.T) {
	s := vramServer(t)
	fb := newFakeBackend(t, 0)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	mw.WriteField("model", "ming")
	mw.Close()
	r := httptest.NewRequest("POST", "/v1/images/jobs", &buf)
	r.Header.Set("Content-Type", mw.FormDataContentType())
	rec := httptest.NewRecorder()
	s.handleImageJobCreate(rec, r)
	if rec.Code != http.StatusBadRequest || !strings.Contains(errCode(t, rec), "image jobs take JSON") {
		t.Errorf("multipart: %d %s", rec.Code, rec.Body.String())
	}
	for body, want := range map[string]string{
		`{"model":"nope","prompt":"x"}`: "model_not_found",
		`not json`:                      "not valid JSON",
	} {
		if rec := postImageJob(s, body); !strings.Contains(errCode(t, rec), want) {
			t.Errorf("%s: %d %s, want %q", body, rec.Code, rec.Body.String(), want)
		}
	}
	rec = httptest.NewRecorder()
	s.handleImageJobGet(rec, withID(httptest.NewRequest("GET", "/", nil), "0000000000000000"))
	if rec.Code != http.StatusNotFound || !strings.Contains(errCode(t, rec), "job_not_found") {
		t.Errorf("unknown job: %d %s", rec.Code, rec.Body.String())
	}
	s.cfg.GenerationEnabled = false
	if rec := postImageJob(s, `{"model":"ming","prompt":"x"}`); rec.Code != http.StatusNotFound {
		t.Errorf("with generation off: %d", rec.Code)
	}
}

// A backend error fails the job with the step and the backend's message.
func TestImageJobBackendError(t *testing.T) {
	s := vramServer(t)
	fb := &fakeBackend{srv: httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			return
		}
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":{"message":"invalid size"}}`))
	}))}
	t.Cleanup(fb.srv.Close)
	runFakeInstance(t, s, "ming", "ming-image", 0, fb)
	id := createdID(t, postImageJob(s, `{"model":"ming","prompt":"x","size":"7x7"}`))
	if j := waitImageJob(t, s, id, jobFailed); j.Error != "generating: HTTP 400: invalid size" {
		t.Errorf("error = %q", j.Error)
	}
}

// The image job routes need the API key and sit before the LLM proxy's
// catch-all.
func TestImageJobRoutesRegistered(t *testing.T) {
	s := vramServer(t)
	s.cfg.APIKey = "secret"
	router := s.buildRouter()
	for _, rq := range []*http.Request{
		httptest.NewRequest("POST", "/v1/images/jobs", strings.NewReader(`{}`)),
		httptest.NewRequest("GET", "/v1/images/jobs/abc", nil),
		httptest.NewRequest("GET", "/v1/images/jobs/abc/image/0", nil),
		httptest.NewRequest("DELETE", "/v1/images/jobs/abc", nil),
	} {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, rq)
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("%s %s without a key: %d, want 401", rq.Method, rq.URL.Path, rec.Code)
		}
		rq.Header.Set("Authorization", "Bearer secret")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, rq)
		if !strings.Contains(rec.Body.String(), "job_not_found") && !strings.Contains(rec.Body.String(), "model_not_found") {
			t.Errorf("%s %s did not reach the image job handler: %d %s", rq.Method, rq.URL.Path, rec.Code, rec.Body.String())
		}
	}
}

func TestImageJobsPrune(t *testing.T) {
	s := vramServer(t)
	dir := t.TempDir()
	old, recent := time.Now().Add(-25*time.Hour), time.Now()
	for i, fin := range []*time.Time{&old, &recent, nil} {
		id := fmt.Sprintf("%016x", i)
		var paths []string
		for n := range 2 {
			file := filepath.Join(dir, fmt.Sprintf("%s-%d.png", id, n))
			os.WriteFile(file, []byte("x"), 0o644)
			paths = append(paths, file)
		}
		s.genImageJobs.add(&imageJob{genJob: genJob{ID: id, Status: jobSucceeded, Finished: fin}, paths: paths})
	}
	s.genImageJobs.prune(time.Now())
	if _, ok := s.genImageJobs.view("0000000000000000"); ok {
		t.Error("a job finished 25 hours ago was kept")
	}
	if files, _ := filepath.Glob(filepath.Join(dir, "0000000000000000-*")); len(files) != 0 {
		t.Errorf("its image files were kept: %v", files)
	}
	for _, id := range []string{"0000000000000001", "0000000000000002"} {
		if _, ok := s.genImageJobs.view(id); !ok {
			t.Errorf("job %s was pruned", id)
		}
	}
}

// A job for a stopped backend, with eviction on or off, says it is
// starting the backend, and fails with the reason when the backend cannot
// start.
func TestImageJobStartingPhase(t *testing.T) {
	for _, evict := range []bool{false, true} {
		s := vramServer(t)
		s.cfg.GenEvict = evict
		addInstance(t, s, generation.Instance{ID: "ming", Port: 8089, Enabled: true, ReservedMiB: 4096})
		rec := postImageJob(s, `{"model":"ming","prompt":"x"}`)
		var created map[string]any
		json.Unmarshal(rec.Body.Bytes(), &created)
		if rec.Code != http.StatusAccepted || created["status"] != "running" || created["phase"] != "starting" {
			t.Fatalf("eviction %v: create: %d %s", evict, rec.Code, rec.Body.String())
		}
		// The test build's binary does not exist.
		if j := waitImageJob(t, s, created["id"].(string), jobFailed); !strings.HasPrefix(j.Error, "starting the backend: ") {
			t.Errorf("eviction %v: error = %q", evict, j.Error)
		}
	}
}
