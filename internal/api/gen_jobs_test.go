package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

// openStore opens a store on dir's records and outputs, as a restart does.
func openStore[T jobRecord[T]](t *testing.T, dir string, now time.Time) *jobStore[T] {
	t.Helper()
	st := &jobStore[T]{}
	if err := st.open(filepath.Join(dir, "jobs"), filepath.Join(dir, "out"), now); err != nil {
		t.Fatal(err)
	}
	return st
}

func writeOutput(t *testing.T, dir, name string) string {
	t.Helper()
	file := filepath.Join(dir, "out", name)
	os.MkdirAll(filepath.Dir(file), 0o755)
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	return file
}

func TestJobStoreKeepsJobsAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	start := time.Now().Add(-time.Hour).Truncate(time.Second)
	st := openStore[imageJob](t, dir, start)
	done := start.Add(time.Minute)
	file := writeOutput(t, dir, "aaaaaaaaaaaaaaaa-0.png")
	st.add(&imageJob{genJob: genJob{ID: "aaaaaaaaaaaaaaaa", Status: jobQueued, Submitted: start}, Request: imageJobRequest{Prompt: "a fox"}})
	st.update("aaaaaaaaaaaaaaaa", func(j *imageJob) {
		j.Status, j.Finished, j.paths = jobSucceeded, &done, []string{file}
		j.Images = []imageJobImage{{URL: "/v1/images/jobs/aaaaaaaaaaaaaaaa/image/0"}}
	})
	st.add(&imageJob{genJob: genJob{ID: "bbbbbbbbbbbbbbbb", Status: jobRunning, Phase: imagePhaseGenerating, Submitted: start.Add(time.Second)}})

	restart := time.Now().Truncate(time.Second)
	st = openStore[imageJob](t, dir, restart)
	j, ok := st.view("aaaaaaaaaaaaaaaa")
	if !ok || j.Status != jobSucceeded || !slices.Equal(j.paths, []string{file}) || len(j.Images) != 1 || j.Request.Prompt != "a fox" {
		t.Errorf("finished job after restart = %+v", j)
	}
	j, ok = st.view("bbbbbbbbbbbbbbbb")
	if !ok || j.Status != jobFailed || j.Phase != "" || j.Error != jobRestartedError || j.Finished == nil || !j.Finished.Equal(restart) {
		t.Errorf("running job after restart = %+v", j)
	}
	if ids := jobIDs(st.list(nil)); !slices.Equal(ids, []string{"bbbbbbbbbbbbbbbb", "aaaaaaaaaaaaaaaa"}) {
		t.Errorf("list = %v, want newest first", ids)
	}

	// The failure was saved: a second restart keeps its time.
	st = openStore[imageJob](t, dir, restart.Add(time.Hour))
	if j, _ := st.view("bbbbbbbbbbbbbbbb"); j.Finished == nil || !j.Finished.Equal(restart) {
		t.Errorf("after a second restart the job finished at %v, want %v", j.Finished, restart)
	}
}

// A job that fails because shutdown stopped its backend is not saved as
// failed with that error: the next start fails it as interrupted.
func TestJobStoreClosedKeepsJobsUnfinished(t *testing.T) {
	dir := t.TempDir()
	st := openStore[imageJob](t, dir, time.Now())
	st.add(&imageJob{genJob: genJob{ID: "aaaaaaaaaaaaaaaa", Status: jobRunning}})
	st.close()
	now := time.Now()
	st.update("aaaaaaaaaaaaaaaa", func(j *imageJob) { j.Status, j.Error, j.Finished = jobFailed, "generating: EOF", &now })
	if j, _ := openStore[imageJob](t, dir, time.Now()).view("aaaaaaaaaaaaaaaa"); j.Error != jobRestartedError {
		t.Errorf("after a restart the job's error is %q, want %q", j.Error, jobRestartedError)
	}
}

func TestJobStoreRestoresPrivateParts(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	music := openStore[musicJob](t, dir, now)
	music.add(&musicJob{genJob: genJob{ID: "m", Status: musicSucceeded, Finished: &now}, audioPath: "/x/m.mp3", lmModel: "lm", ditModel: "dit"})
	if j, _ := openStore[musicJob](t, dir, now).view("m"); j.audioPath != "/x/m.mp3" || j.lmModel != "lm" || j.ditModel != "dit" {
		t.Errorf("music job after restart = %+v", j)
	}

	dir = t.TempDir()
	video := openStore[videoJob](t, dir, now)
	video.add(&videoJob{genJob: genJob{ID: "v", Status: jobQueued}, done: make(chan struct{})})
	j, _ := openStore[videoJob](t, dir, now).view("v")
	if j.Status != jobFailed {
		t.Errorf("video job after restart is %q, want failed", j.Status)
	}
	select {
	case <-j.done:
	default:
		t.Error("a restored video job's done channel is open, so the waiting route would hang")
	}
}

func TestJobStoreListFilters(t *testing.T) {
	st := &jobStore[imageJob]{}
	now := time.Now()
	st.add(&imageJob{genJob: genJob{ID: "1", Status: jobSucceeded, Finished: &now}})
	st.add(&imageJob{genJob: genJob{ID: "2", Status: jobQueued}})
	st.add(&imageJob{genJob: genJob{ID: "3", Status: jobFailed, Finished: &now}})
	if ids := jobIDs(st.list([]string{jobSucceeded, jobFailed})); !slices.Equal(ids, []string{"3", "1"}) {
		t.Errorf("list(succeeded,failed) = %v", ids)
	}
	if _, ok := jobStatusFilter("queued,done"); ok {
		t.Error("an unknown status was accepted")
	}
	if got, ok := jobStatusFilter("queued, running"); !ok || !slices.Equal(got, []string{jobQueued, jobRunning}) {
		t.Errorf("jobStatusFilter = %v %v", got, ok)
	}
}

func TestJobStoreRemove(t *testing.T) {
	dir := t.TempDir()
	st := openStore[imageJob](t, dir, time.Now())
	now := time.Now()
	file := writeOutput(t, dir, "aaaaaaaaaaaaaaaa-0.png")
	st.add(&imageJob{genJob: genJob{ID: "aaaaaaaaaaaaaaaa", Status: jobSucceeded, Finished: &now}, paths: []string{file}})
	st.add(&imageJob{genJob: genJob{ID: "bbbbbbbbbbbbbbbb", Status: jobRunning}})

	if st.remove("bbbbbbbbbbbbbbbb") {
		t.Error("a running job was removed")
	}
	if !st.remove("aaaaaaaaaaaaaaaa") {
		t.Fatal("a finished job was not removed")
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("its image was kept")
	}
	if _, err := os.Stat(st.recordPath("aaaaaaaaaaaaaaaa")); !os.IsNotExist(err) {
		t.Error("its record was kept")
	}
	if ids := jobIDs(st.list(nil)); !slices.Equal(ids, []string{"bbbbbbbbbbbbbbbb"}) {
		t.Errorf("list = %v", ids)
	}
	if _, ok := openStore[imageJob](t, dir, time.Now()).view("aaaaaaaaaaaaaaaa"); ok {
		t.Error("the removed job came back after a restart")
	}
}

// Output files no job owns, such as those from before records were kept,
// are removed once they are older than jobKeep.
func TestJobStorePrunesOrphanFiles(t *testing.T) {
	dir := t.TempDir()
	st := openStore[musicJob](t, dir, time.Now())
	old := time.Now().Add(-jobKeep - time.Hour)
	owned := writeOutput(t, dir, "aaaaaaaaaaaaaaaa.mp3")
	orphan := writeOutput(t, dir, "bbbbbbbbbbbbbbbb.mp3")
	fresh := writeOutput(t, dir, "cccccccccccccccc.wav")
	for _, f := range []string{owned, orphan} {
		os.Chtimes(f, old, old)
	}
	st.add(&musicJob{genJob: genJob{ID: "aaaaaaaaaaaaaaaa", Status: musicRunning}, audioPath: owned})

	st.prune(time.Now())
	if _, err := os.Stat(orphan); !os.IsNotExist(err) {
		t.Error("an old file no job owns was kept")
	}
	for _, f := range []string{owned, fresh} {
		if _, err := os.Stat(f); err != nil {
			t.Errorf("%s was removed: %v", filepath.Base(f), err)
		}
	}
}

func TestImageJobListAndDeleteRoutes(t *testing.T) {
	s := vramServer(t)
	s.openGenJobs()
	now := time.Now()
	file := filepath.Join(s.imageDir(), "aaaaaaaaaaaaaaaa-0.png")
	os.MkdirAll(s.imageDir(), 0o755)
	os.WriteFile(file, []byte("img"), 0o644)
	s.genImageJobs.add(&imageJob{genJob: genJob{ID: "aaaaaaaaaaaaaaaa", Status: jobSucceeded, Finished: &now}, paths: []string{file}})

	list := func(q string) (int, []imageJob) {
		rec := httptest.NewRecorder()
		s.handleImageJobList(rec, httptest.NewRequest("GET", "/v1/images/jobs"+q, nil))
		var out jobListResponse[imageJob]
		json.Unmarshal(rec.Body.Bytes(), &out)
		return rec.Code, out.Data
	}
	if code, jobs := list("?status=succeeded"); code != http.StatusOK || len(jobs) != 1 {
		t.Errorf("list: %d %+v", code, jobs)
	}
	if code, jobs := list("?status=queued"); code != http.StatusOK || len(jobs) != 0 {
		t.Errorf("list queued: %d %+v", code, jobs)
	}
	if code, _ := list("?status=done"); code != http.StatusBadRequest {
		t.Errorf("list with an unknown status: %d, want 400", code)
	}

	rec := httptest.NewRecorder()
	s.handleImageJobDelete(rec, withID(httptest.NewRequest("DELETE", "/", nil), "aaaaaaaaaaaaaaaa"))
	if rec.Code != http.StatusNoContent {
		t.Fatalf("delete: %d %s", rec.Code, rec.Body)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("the image was kept")
	}
	rec = httptest.NewRecorder()
	s.handleImageJobGet(rec, withID(httptest.NewRequest("GET", "/", nil), "aaaaaaaaaaaaaaaa"))
	if rec.Code != http.StatusNotFound {
		t.Errorf("get after delete: %d, want 404", rec.Code)
	}
}

// The list routes are behind the API key, like the job routes.
func TestJobListRoutesNeedTheKey(t *testing.T) {
	s, _ := musicServer(t)
	s.cfg.APIKey = "secret"
	router := s.buildRouter()
	for _, path := range []string{"/v1/images/jobs", "/v1/music/jobs", "/v1/videos/jobs"} {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		if rec.Code != http.StatusUnauthorized {
			t.Errorf("GET %s without a key: %d, want 401", path, rec.Code)
		}
		rq := httptest.NewRequest("GET", path, nil)
		rq.Header.Set("Authorization", "Bearer secret")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, rq)
		if rec.Code == http.StatusUnauthorized || rec.Code == http.StatusMethodNotAllowed {
			t.Errorf("GET %s with the key: %d %s", path, rec.Code, rec.Body)
		}
	}
}

func jobIDs[T jobRecord[T]](jobs []T) []string {
	var ids []string
	for _, j := range jobs {
		ids = append(ids, j.state().id)
	}
	return ids
}

// start leaves a cancelled job alone, and sets Started only the first time.
func TestGenJobStart(t *testing.T) {
	first := time.Unix(100, 0)
	g := genJob{Status: jobQueued}
	if !g.start(imagePhaseStarting, first) || g.Status != jobRunning || g.Phase != imagePhaseStarting || !g.Started.Equal(first) {
		t.Fatalf("queued job: %+v", g)
	}
	if !g.start(imagePhaseGenerating, first.Add(time.Minute)) || g.Phase != imagePhaseGenerating || !g.Started.Equal(first) {
		t.Fatalf("second start moved Started or kept the phase: %+v", g)
	}
	c := genJob{Status: jobCancelled}
	if c.start(imagePhaseGenerating, first) || c.Status != jobCancelled || c.Started != nil {
		t.Fatalf("cancelled job was started: %+v", c)
	}
}
