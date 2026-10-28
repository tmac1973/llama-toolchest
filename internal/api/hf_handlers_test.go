package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/tmac1973/llama-toolchest/internal/config"
	"github.com/tmac1973/llama-toolchest/internal/huggingface"
	"github.com/tmac1973/llama-toolchest/internal/models"
	"github.com/tmac1973/llama-toolchest/internal/modelsource"
	"github.com/tmac1973/llama-toolchest/internal/testutil"
)

// hfhServer returns a Server with live HuggingFace and ModelScope clients
// and a downloader, as the HuggingFace handlers need.
func hfhServer(t *testing.T) *Server {
	t.Helper()
	return newSettingsServer(t, &config.Config{ExternalURL: "http://localhost:3000"})
}

// hfhModelScope points the server's ModelScope client at a fake API and
// returns a counter of the requests it received. The HuggingFace client
// cannot be pointed at a fake from this package (its fields are
// unexported), so tests that need an upstream answer use the ModelScope
// source. The handlers treat both sources the same way.
func hfhModelScope(t *testing.T, s *Server, handler http.HandlerFunc) *atomic.Int32 {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		handler(w, r)
	}))
	t.Cleanup(srv.Close)
	s.msClient.SetHTTPClientForTest(srv.Client(), srv.URL)
	return &hits
}

const hfhMSSearchBody = `{"Code":200,"Success":true,"Data":{"Model":{"Models":[
	{"Name":"Qwen3-8B-GGUF","Path":"unsloth","Downloads":5000,"Stars":12,"Libraries":["gguf"]}]}}}`

const hfhMSFilesBody = `{"Code":200,"Success":true,"Data":{"Files":[
	{"Name":"Qwen3-8B-Q4_K_M.gguf","Path":"Qwen3-8B-Q4_K_M.gguf","Size":5000000000,"Type":"blob"},
	{"Name":"README.md","Path":"README.md","Size":10,"Type":"blob"}]}}`

func hfhGet(s *Server, h http.HandlerFunc, target string, htmx bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, target, nil)
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	h(rec, req)
	return rec
}

// A search without a query is refused before any request goes upstream.
func TestHFSearchRequiresQuery(t *testing.T) {
	s := hfhServer(t)
	hits := hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) {})
	rec := hfhGet(s, s.handleHFSearch, "/api/hf/search?source=modelscope", false)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "missing q") {
		t.Errorf("status %d body %q, want 400 naming the missing q", rec.Code, rec.Body.String())
	}
	if hits.Load() != 0 {
		t.Errorf("%d upstream requests for a refused search", hits.Load())
	}
}

// An upstream failure is a 502 with the upstream status as plain text,
// for htmx requests too. The page's error notice (web/static/errors.js)
// shows the text of a non-2xx answer, so a 401 from an expired token is
// visible instead of an empty result list.
func TestHFSearchUpstreamErrorIsBadGateway(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusNotFound, http.StatusInternalServerError} {
		s := hfhServer(t)
		hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(code) })
		for _, htmx := range []bool{false, true} {
			rec := hfhGet(s, s.handleHFSearch, "/api/hf/search?q=qwen&source=modelscope", htmx)
			if rec.Code != http.StatusBadGateway {
				t.Errorf("upstream %d, htmx %v: status %d, want 502", code, htmx, rec.Code)
			}
			if body := rec.Body.String(); !strings.Contains(body, "HTTP "+strconv.Itoa(code)) {
				t.Errorf("upstream %d, htmx %v: body %q does not name the upstream status", code, htmx, body)
			}
			if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
				t.Errorf("upstream %d, htmx %v: Content-Type %q, want plain text for the error notice", code, htmx, ct)
			}
		}
	}
}

// A JSON caller gets the results as JSON; an htmx caller gets the result
// rows, whose Files button carries the source the result came from.
func TestHFSearchAnswersJSONAndHTML(t *testing.T) {
	s := hfhServer(t)
	hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(hfhMSSearchBody)) })

	rec := hfhGet(s, s.handleHFSearch, "/api/hf/search?q=qwen&source=modelscope", false)
	if rec.Code != http.StatusOK || !strings.HasPrefix(rec.Header().Get("Content-Type"), "application/json") {
		t.Fatalf("JSON: status %d, Content-Type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	var got []modelsource.SearchResult
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("JSON body: %v\n%s", err, rec.Body.String())
	}
	if len(got) != 1 || got[0].ID != "unsloth/Qwen3-8B-GGUF" || got[0].Downloads != 5000 {
		t.Errorf("results = %+v", got)
	}

	rec = hfhGet(s, s.handleHFSearch, "/api/hf/search?q=qwen&source=modelscope", true)
	body := rec.Body.String()
	if rec.Code != http.StatusOK || !strings.HasPrefix(rec.Header().Get("Content-Type"), "text/html") {
		t.Fatalf("htmx: status %d, Content-Type %q", rec.Code, rec.Header().Get("Content-Type"))
	}
	if !strings.Contains(body, "unsloth/Qwen3-8B-GGUF") || !strings.Contains(body, "source=modelscope") {
		t.Errorf("htmx rows lack the result or its source:\n%s", body)
	}
}

// With no source in the request, the search goes to the configured
// default source, which is what the Settings choice is for.
func TestHFSearchFollowsTheDefaultSource(t *testing.T) {
	s := hfhServer(t)
	s.cfg.DefaultModelSource = modelsource.SourceModelScope
	hits := hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(hfhMSSearchBody)) })

	rec := hfhGet(s, s.handleHFSearch, "/api/hf/search?q=qwen", false)
	if rec.Code != http.StatusOK || hits.Load() != 1 {
		t.Errorf("status %d, %d ModelScope requests; want 200 from the configured default", rec.Code, hits.Load())
	}
}

// A file list without an id is refused.
func TestHFModelRequiresID(t *testing.T) {
	s := hfhServer(t)
	rec := hfhGet(s, s.handleHFModel, "/api/hf/model?source=modelscope", true)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "missing id") {
		t.Errorf("status %d body %q, want 400 naming the missing id", rec.Code, rec.Body.String())
	}
}

// An upstream failure listing a repository is a 502 that names the
// upstream status, as for search.
func TestHFModelUpstreamErrorIsBadGateway(t *testing.T) {
	for _, code := range []int{http.StatusUnauthorized, http.StatusNotFound, http.StatusInternalServerError} {
		s := hfhServer(t)
		hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(code) })
		rec := hfhGet(s, s.handleHFModel, "/api/hf/model?id=unsloth/Qwen3-8B-GGUF&source=modelscope", true)
		if rec.Code != http.StatusBadGateway || !strings.Contains(rec.Body.String(), "HTTP "+strconv.Itoa(code)) {
			t.Errorf("upstream %d: status %d body %q, want 502 naming the status", code, rec.Code, rec.Body.String())
		}
	}
}

// A failed listing is not cached: the next request asks upstream again,
// so a repository that was briefly unavailable can be opened once it is
// back, without waiting for the cache to expire.
func TestHFModelDoesNotCacheAFailure(t *testing.T) {
	s := hfhServer(t)
	var fail atomic.Bool
	fail.Store(true)
	hits := hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) {
		if fail.Load() {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Write([]byte(hfhMSFilesBody))
	})
	const target = "/api/hf/model?id=unsloth/Qwen3-8B-GGUF&source=modelscope"
	if rec := hfhGet(s, s.handleHFModel, target, false); rec.Code != http.StatusBadGateway {
		t.Fatalf("first request: status %d, want 502", rec.Code)
	}
	fail.Store(false)
	if rec := hfhGet(s, s.handleHFModel, target, false); rec.Code != http.StatusOK {
		t.Fatalf("second request: status %d, want 200 once upstream works", rec.Code)
	}
	if hits.Load() != 2 {
		t.Errorf("%d upstream requests, want 2", hits.Load())
	}
}

// The file list and the deferred estimates request both read the
// repository; the short cache means one click costs one upstream listing,
// which matters on a host that limits request rates.
func TestHFModelCachesTheListingBriefly(t *testing.T) {
	s := hfhServer(t)
	hits := hfhModelScope(t, s, func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(hfhMSFilesBody)) })
	const target = "/api/hf/model?id=unsloth/Qwen3-8B-GGUF&source=modelscope"
	for i := range 3 {
		if rec := hfhGet(s, s.handleHFModel, target, true); rec.Code != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, rec.Code, rec.Body.String())
		}
	}
	if hits.Load() != 1 {
		t.Errorf("%d upstream listings for three requests, want 1", hits.Load())
	}
}

// hfhSeedDetail puts a repository listing in the server's short cache, so
// handleHFModel can be tested with the HuggingFace source without any
// request leaving the test.
func hfhSeedDetail(s *Server, source string, d *modelsource.Detail) {
	s.detailMu.Lock()
	defer s.detailMu.Unlock()
	if s.detailCache == nil {
		s.detailCache = map[string]detailEntry{}
	}
	s.detailCache[source+"\x00"+d.ID] = detailEntry{detail: d, at: time.Now()}
}

// The JSON view marks a file that is already registered as downloaded,
// and a file larger than the free disk as not fitting. A file of unknown
// size (0) is allowed to try, since the size is missing, not too large.
func TestHFModelMarksDownloadedAndFit(t *testing.T) {
	s := hfhServer(t)
	if s.downloader.AvailableForDownload() < 0 {
		t.Skip("free disk space cannot be read here")
	}
	const repo = "org/Model-GGUF"
	hfhSeedDetail(s, modelsource.SourceHuggingFace, &modelsource.Detail{ID: repo, Files: []modelsource.File{
		{Filename: "have.gguf", Size: 100},
		{Filename: "huge.gguf", Size: 1 << 60},
		{Filename: "unknown.gguf", Size: 0},
	}})
	if err := s.registry.Add(&models.Model{ID: "org--Model-GGUF--have", ModelID: repo, Filename: "have.gguf"}); err != nil {
		t.Fatal(err)
	}

	rec := hfhGet(s, s.handleHFModel, "/api/hf/model?id="+repo+"&source=hf", false)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var view struct {
		ID     string
		Source string
		Files  []struct {
			Filename          string `json:"filename"`
			AlreadyDownloaded bool
			FitsOnDisk        bool
		}
		SafetyMargin int64
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &view); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if view.ID != repo || view.Source != modelsource.SourceHuggingFace || view.SafetyMargin != huggingface.DiskSafetyMarginBytes {
		t.Errorf("view = id %q source %q margin %d", view.ID, view.Source, view.SafetyMargin)
	}
	got := map[string][2]bool{}
	for _, f := range view.Files {
		got[f.Filename] = [2]bool{f.AlreadyDownloaded, f.FitsOnDisk}
	}
	want := map[string][2]bool{
		"have.gguf":    {true, false}, // downloaded files are not offered again
		"huge.gguf":    {false, false},
		"unknown.gguf": {false, true},
	}
	for name, w := range want {
		if got[name] != w {
			t.Errorf("%s: (downloaded, fits) = %v, want %v", name, got[name], w)
		}
	}
}

// A panel opened from a recommendation names the suggested file and the
// other repositories, leaving out the one already shown and blank
// entries. An unknown context name is ignored rather than shown.
func TestHFModelRecommendationFields(t *testing.T) {
	s := hfhServer(t)
	const repo = "org/Model-GGUF"
	hfhSeedDetail(s, modelsource.SourceHuggingFace, &modelsource.Detail{ID: repo, Files: []modelsource.File{{Filename: "m.gguf", Size: 100}}})

	decode := func(target string) hfModelView {
		t.Helper()
		rec := hfhGet(s, s.handleHFModel, target, false)
		var v hfModelView
		if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return v
	}
	q := url.Values{"id": {repo}, "source": {"hf"}, "ctx": {"32k"}, "suggest": {"m.gguf"},
		"alts": {"other/A-GGUF, ," + repo + ",other/B-GGUF"}}
	v := decode("/api/hf/model?" + q.Encode())
	if v.Ctx != "32k" || v.Suggest != "m.gguf" || !strings.HasPrefix(v.SuggestLabel, "Suggested for ") {
		t.Errorf("ctx %q suggest %q label %q", v.Ctx, v.Suggest, v.SuggestLabel)
	}
	if len(v.Alts) != 2 || v.Alts[0] != "other/A-GGUF" || v.Alts[1] != "other/B-GGUF" {
		t.Errorf("alts = %q, want the two other repositories only", v.Alts)
	}

	q.Set("ctx", "max")
	if v := decode("/api/hf/model?" + q.Encode()); v.SuggestLabel != "Suggested for the model's maximum context" {
		t.Errorf("max label = %q", v.SuggestLabel)
	}

	q.Set("ctx", "999k")
	if v := decode("/api/hf/model?" + q.Encode()); v.Ctx != "" || v.Suggest != "" || v.Alts != nil {
		t.Errorf("unknown ctx kept: ctx %q suggest %q alts %q", v.Ctx, v.Suggest, v.Alts)
	}
}

// hfhFileHost is a fake download host. Each GET sends the first bytes and
// then waits until release is closed (or the client leaves), so a
// download stays in progress for as long as a test needs.
type hfhFileHost struct {
	srv     *httptest.Server
	release chan struct{}
	gets    atomic.Int32
	once    sync.Once
}

func newHFHFileHost(t *testing.T, body string, block bool) *hfhFileHost {
	t.Helper()
	h := &hfhFileHost{release: make(chan struct{})}
	if !block {
		h.unblock()
	}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			h.gets.Add(1)
		}
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		if r.Method == http.MethodHead {
			return
		}
		w.Write([]byte(body[:1]))
		w.(http.Flusher).Flush()
		select {
		case <-h.release:
			w.Write([]byte(body[1:]))
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() {
		h.unblock()
		h.srv.Close()
	})
	return h
}

func (h *hfhFileHost) unblock() { h.once.Do(func() { close(h.release) }) }

// hfhUseFileHost makes HuggingFace downloads fetch from the fake host.
func hfhUseFileHost(s *Server, h *hfhFileHost) {
	s.downloader.RegisterProvider(modelsource.SourceHuggingFace, huggingface.Provider{
		URL: func(modelID, filename string) string { return h.srv.URL + "/" + modelID + "/" + filename },
	})
}

func hfhPost(s *Server, form url.Values, htmx bool) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/api/hf/download", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	s.handleHFDownload(rec, req)
	return rec
}

func hfhCancel(s *Server, id string) *httptest.ResponseRecorder {
	r := chi.NewRouter()
	r.Delete("/api/hf/download/{id}", s.handleHFDownloadCancel)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/api/hf/download/"+id, nil))
	return rec
}

// A JSON download request answers 202 with the download id, and the file
// arrives in the models directory under the repository's folder.
func TestHFDownloadJSONStartsTheDownload(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "GGUF-bytes", false)
	hfhUseFileHost(s, host)

	req := httptest.NewRequest(http.MethodPost, "/api/hf/download",
		strings.NewReader(`{"model_id":"org/Model-GGUF","filename":"m.gguf","size":10}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleHFDownload(rec, req)

	if rec.Code != http.StatusAccepted {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	var resp map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v\n%s", err, rec.Body.String())
	}
	if resp["download_id"] != "org--Model-GGUF--m" {
		t.Errorf("download_id = %q", resp["download_id"])
	}
	path := filepath.Join(s.cfg.DataDir, "org--Model-GGUF", "m.gguf")
	testutil.Eventually(t, func() bool {
		data, err := os.ReadFile(path)
		return err == nil && string(data) == "GGUF-bytes"
	}, "downloaded file %s never appeared with the served bytes", path)
}

// The 202 answer must say it is JSON. Setting the Content-Type after
// WriteHeader has sent the headers would lose it, and the body would be
// sent as text/plain.
func TestHFDownloadJSONAnswerHasJSONContentType(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "GGUF-bytes", false)
	hfhUseFileHost(s, host)
	srv := httptest.NewServer(http.HandlerFunc(s.handleHFDownload))
	defer srv.Close()

	resp, err := http.Post(srv.URL, "application/json", strings.NewReader(`{"model_id":"org/Model-GGUF","filename":"ct.gguf"}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	s.downloader.Cancel("org--Model-GGUF--ct")
}

// A body that is not valid JSON is refused with 400 and starts nothing.
func TestHFDownloadRejectsBadJSON(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "x", false)
	hfhUseFileHost(s, host)
	req := httptest.NewRequest(http.MethodPost, "/api/hf/download", strings.NewReader(`{"model_id":`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleHFDownload(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status %d, want 400", rec.Code)
	}
	if active := s.downloader.ListActive(); len(active) != 0 || host.gets.Load() != 0 {
		t.Errorf("a refused request started downloads: %+v", active)
	}
}

// A file larger than the free disk is refused before anything starts:
// 507 with the reason for a normal request, which the page's error
// notice shows. An inline htmx request (restore report rows, pending
// cards) gets a 200 with the reason marked, because htmx does not swap a
// non-2xx answer into those rows.
func TestHFDownloadRefusesAFileThatDoesNotFit(t *testing.T) {
	s := hfhServer(t)
	if s.downloader.AvailableForDownload() < 0 {
		t.Skip("free disk space cannot be read here")
	}
	host := newHFHFileHost(t, "x", false)
	hfhUseFileHost(s, host)
	form := url.Values{"model_id": {"org/Huge-GGUF"}, "filename": {"huge.gguf"}, "size": {"1152921504606846976"}} // 1 EiB

	rec := hfhPost(s, form, true)
	if rec.Code != http.StatusInsufficientStorage || !strings.Contains(rec.Body.String(), "insufficient disk space") {
		t.Errorf("htmx: status %d body %q, want 507 with the reason", rec.Code, rec.Body.String())
	}

	form.Set("inline", "1")
	rec = hfhPost(s, form, true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "<mark>insufficient disk space") {
		t.Errorf("inline htmx: status %d body %q, want 200 with a marked reason", rec.Code, rec.Body.String())
	}

	// Inline only applies to htmx: a plain caller still gets the status.
	rec = hfhPost(s, form, false)
	if rec.Code != http.StatusInsufficientStorage {
		t.Errorf("inline without htmx: status %d, want 507", rec.Code)
	}
	if host.gets.Load() != 0 || len(s.downloader.ListActive()) != 0 {
		t.Error("a refused download was started")
	}
}

// Starting a download that is already running is refused (400, or a
// marked 200 for an inline htmx row). The running download is then
// paused with DELETE, which answers 204; pausing it again, or pausing an
// id that never existed, answers 404.
func TestHFDownloadDuplicateAndCancel(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "GGUF-bytes", true)
	hfhUseFileHost(s, host)
	form := url.Values{"model_id": {"org/Model-GGUF"}, "filename": {"m.gguf"}}

	rec := hfhPost(s, form, true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `hx-get="/api/hf/download/org--Model-GGUF--m/progress"`) {
		t.Fatalf("first start: status %d, want the progress row:\n%s", rec.Code, rec.Body.String())
	}
	testutil.Eventually(t, func() bool { return host.gets.Load() == 1 }, "the download never reached the host")

	rec = hfhPost(s, form, true)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "already in progress") {
		t.Errorf("second start: status %d body %q, want 400 already in progress", rec.Code, rec.Body.String())
	}
	form.Set("inline", "1")
	rec = hfhPost(s, form, true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "<mark>download already in progress</mark>") {
		t.Errorf("inline second start: status %d body %q, want 200 with a marked error", rec.Code, rec.Body.String())
	}

	if rec := hfhCancel(s, "org--Model-GGUF--m"); rec.Code != http.StatusNoContent {
		t.Errorf("pause: status %d body %q, want 204", rec.Code, rec.Body.String())
	}
	if st, ok := s.downloader.Status("org--Model-GGUF--m"); !ok || st.Status != "cancelled" {
		t.Errorf("after pause: status %+v (held %v), want cancelled", st, ok)
	}
	if _, err := os.Stat(filepath.Join(s.cfg.DataDir, "org--Model-GGUF", "m.gguf.part")); err != nil {
		t.Errorf("pause must keep the partial file for a resume: %v", err)
	}

	if rec := hfhCancel(s, "no--such--download"); rec.Code != http.StatusNotFound || !strings.Contains(rec.Body.String(), "no active download") {
		t.Errorf("unknown id: status %d body %q, want 404", rec.Code, rec.Body.String())
	}
}

// An inline htmx start answers with a short text line, and a start from
// a recommendation remembers the context size, so the finished model
// starts with the settings the card showed. Without from=recommend the
// context size is not remembered.
func TestHFDownloadInlineAndRecommendationSeed(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "GGUF-bytes", true)
	hfhUseFileHost(s, host)

	rec := hfhPost(s, url.Values{"model_id": {"org/Rec-GGUF"}, "filename": {"r.gguf"},
		"ctx": {"128k"}, "from": {"recommend"}, "inline": {"1"}}, true)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "queued") || strings.Contains(rec.Body.String(), "<mark>") {
		t.Errorf("inline start: status %d body %q, want 200 queued", rec.Code, rec.Body.String())
	}
	if class, ok := s.takeSeed("org--Rec-GGUF--r"); !ok || class != models.ContextLong {
		t.Errorf("seed = %v (%v), want the 128k class remembered", class, ok)
	}

	hfhPost(s, url.Values{"model_id": {"org/Plain-GGUF"}, "filename": {"p.gguf"}, "ctx": {"128k"}}, false)
	if _, ok := s.takeSeed("org--Plain-GGUF--p"); ok {
		t.Error("a context size was remembered for a download not started from a recommendation")
	}

	hfhCancel(s, "org--Rec-GGUF--r")
	hfhCancel(s, "org--Plain-GGUF--p")
}

func hfhDiscard(s *Server, q url.Values) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	s.handleIncompleteDiscard(rec, httptest.NewRequest(http.MethodDelete, "/api/hf/incomplete?"+q.Encode(), nil))
	return rec
}

// Discard needs both the repository and the file name; without them it
// cannot know what to delete and must delete nothing.
func TestIncompleteDiscardRequiresBothFields(t *testing.T) {
	s := hfhServer(t)
	for _, q := range []url.Values{{}, {"model_id": {"org/M"}}, {"filename": {"m.gguf"}}} {
		if rec := hfhDiscard(s, q); rec.Code != http.StatusBadRequest {
			t.Errorf("%v: status %d, want 400", q, rec.Code)
		}
	}
}

// Discarding a split download removes every shard and every .part
// fragment of it, and nothing else in the model's folder: another quant
// of the same repository stays.
func TestIncompleteDiscardRemovesEveryShard(t *testing.T) {
	s := hfhServer(t)
	dir := filepath.Join(s.cfg.ModelsPath(), "org--Big-GGUF")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	gone := []string{"Big-00001-of-00003.gguf", "Big-00002-of-00003.gguf.part"}
	kept := []string{"Other-Q8_0.gguf", "Other-Q8_0.gguf.part"}
	for _, n := range append(append([]string{}, gone...), kept...) {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	rec := hfhDiscard(s, url.Values{"model_id": {"org/Big-GGUF"}, "filename": {"Big-00001-of-00003.gguf"}})
	if rec.Code != http.StatusOK || rec.Body.Len() != 0 {
		t.Errorf("status %d body %q, want 200 and an empty body so the row clears", rec.Code, rec.Body.String())
	}
	for _, n := range gone {
		if _, err := os.Stat(filepath.Join(dir, n)); !os.IsNotExist(err) {
			t.Errorf("%s still exists (err %v)", n, err)
		}
	}
	for _, n := range kept {
		if _, err := os.Stat(filepath.Join(dir, n)); err != nil {
			t.Errorf("%s was removed: %v", n, err)
		}
	}
}

// A file name that climbs out of the models directory must not delete
// anything there: the request comes from the browser, and a crafted one
// must not reach other files on the machine.
func TestIncompleteDiscardStaysInsideTheModelsDir(t *testing.T) {
	s := hfhServer(t)
	root := s.cfg.ModelsPath()
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(filepath.Dir(root), "precious.gguf")
	if err := os.WriteFile(outside, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outside+".part", []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	for _, q := range []url.Values{
		{"model_id": {"org/M"}, "filename": {"../../precious.gguf"}},
		{"model_id": {".."}, "filename": {"precious.gguf"}},
	} {
		if rec := hfhDiscard(s, q); rec.Code != http.StatusOK {
			t.Errorf("%v: status %d", q, rec.Code)
		}
	}
	for _, p := range []string{outside, outside + ".part"} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("%s outside the models directory was removed: %v", p, err)
		}
	}
}

// A cached probe result is applied without a new header read, and the
// VRAM estimate then covers only the part that becomes resident. This is
// the correction the probe exists for: a large embedding table that is
// streamed from disk must not count against the card.
func TestProbeStreamedBytesUsesTheCache(t *testing.T) {
	s := hfhServer(t)
	const gib = int64(1) << 30
	const repo = "org/Big-GGUF"
	d := &modelsource.Detail{ID: repo, Files: []modelsource.File{{
		Filename: "big-00001-of-00002.gguf", Size: 40 * gib,
		Shards:     []string{"big-00001-of-00002.gguf", "big-00002-of-00002.gguf"},
		ShardSizes: []int64{20 * gib, 20 * gib},
		VRAMEstGB:  models.EstimateVRAM(40 * gib),
	}}}
	s.storeProbe(modelsource.SourceHuggingFace, repo, "big-00001-of-00002.gguf",
		modelsource.ProbeResult{StreamedBytes: 16 * gib, Probed: true})

	// A cancelled context: if the cache were not used, the probe would
	// fail without a network request and leave the file unmeasured.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.probeStreamedBytes(ctx, modelsource.SourceHuggingFace, d)

	f := d.Files[0]
	if !f.StreamProbed || f.StreamedBytes != 16*gib {
		t.Errorf("file = probed %v streamed %d, want the cached 16 GiB", f.StreamProbed, f.StreamedBytes)
	}
	if want := models.EstimateVRAM(24 * gib); f.VRAMEstGB != want {
		t.Errorf("VRAM estimate %v, want %v for the 24 GiB that stays resident", f.VRAMEstGB, want)
	}
}

// Files the probe does not apply to (single files, and split files whose
// shard sizes are missing) keep their estimate and leave no cache entry,
// so nothing is read for them and nothing wrong is remembered.
func TestProbeStreamedBytesSkipsFilesItCannotMeasure(t *testing.T) {
	s := hfhServer(t)
	const gib = int64(1) << 30
	const repo = "org/Mixed-GGUF"
	d := &modelsource.Detail{ID: repo, Files: []modelsource.File{
		{Filename: "single.gguf", Size: 40 * gib, VRAMEstGB: 41},
		{Filename: "nosizes-00001-of-00002.gguf", Size: 40 * gib, VRAMEstGB: 42,
			Shards: []string{"nosizes-00001-of-00002.gguf", "nosizes-00002-of-00002.gguf"}},
		{Filename: "small-00001-of-00002.gguf", Size: 2 * gib, VRAMEstGB: 43,
			Shards:     []string{"small-00001-of-00002.gguf", "small-00002-of-00002.gguf"},
			ShardSizes: []int64{gib, gib}},
	}}
	s.probeStreamedBytes(context.Background(), modelsource.SourceHuggingFace, d)

	for i, f := range d.Files {
		if f.StreamProbed || f.StreamedBytes != 0 || f.VRAMEstGB != float64(41+i) {
			t.Errorf("%s: probed %v streamed %d estimate %v, want it unchanged", f.Filename, f.StreamProbed, f.StreamedBytes, f.VRAMEstGB)
		}
		if _, ok := s.cachedProbe(modelsource.SourceHuggingFace, repo, f.Filename); ok {
			t.Errorf("%s: a result was cached although nothing was measured", f.Filename)
		}
	}
}

// A probe that could not read any shard (here because the request is
// already cancelled, so no connection is made; in use, a rate limit or a
// dropped network) must not be cached as "no embedding table". The cache
// lives as long as the server, so a cached failure keeps the wrong
// estimate until a restart.
func TestProbeStreamedBytesDoesNotCacheAFailedProbe(t *testing.T) {
	s := hfhServer(t)
	const gib = int64(1) << 30
	const repo = "org/Failed-GGUF"
	d := &modelsource.Detail{ID: repo, Files: []modelsource.File{{
		Filename: "failed-00001-of-00002.gguf", Size: 40 * gib, VRAMEstGB: 43,
		Shards:     []string{"failed-00001-of-00002.gguf", "failed-00002-of-00002.gguf"},
		ShardSizes: []int64{20 * gib, 20 * gib},
	}}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s.probeStreamedBytes(ctx, modelsource.SourceHuggingFace, d)

	if f := d.Files[0]; f.StreamProbed {
		t.Errorf("file marked as measured after a failed probe: %+v", f)
	}
	if r, ok := s.cachedProbe(modelsource.SourceHuggingFace, repo, d.Files[0].Filename); ok {
		t.Errorf("failed probe cached as %+v", r)
	}
}

// The probe cache is keyed by source as well as repository and file: the
// same name on two hosts can be different uploads.
func TestProbeCacheIsKeyedBySource(t *testing.T) {
	s := hfhServer(t)
	s.storeProbe(modelsource.SourceHuggingFace, "org/M", "m.gguf", modelsource.ProbeResult{StreamedBytes: 5, Probed: true})
	if _, ok := s.cachedProbe(modelsource.SourceModelScope, "org/M", "m.gguf"); ok {
		t.Error("a HuggingFace probe answered for ModelScope")
	}
	if r, ok := s.cachedProbe(modelsource.SourceHuggingFace, "org/M", "m.gguf"); !ok || r.StreamedBytes != 5 {
		t.Errorf("cached = %+v (%v), want the stored result", r, ok)
	}
}

func hfhProgress(s *Server, id string, htmx bool) *httptest.ResponseRecorder {
	r := chi.NewRouter()
	r.Get("/api/hf/download/{id}/progress", s.handleHFDownloadProgress)
	req := httptest.NewRequest(http.MethodGet, "/api/hf/download/"+id+"/progress", nil)
	if htmx {
		req.Header.Set("HX-Request", "true")
	}
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	return rec
}

// A download row on a page asks for its progress every few seconds instead
// of holding a stream open. It is told to stop asking (status 286) once the
// download has ended, or when the download is no longer known.
func TestHFDownloadProgressPollingStopsWhenTheDownloadEnds(t *testing.T) {
	s := hfhServer(t)
	host := newHFHFileHost(t, "GGUF-bytes", true)
	hfhUseFileHost(s, host)
	form := url.Values{"model_id": {"org/Model-GGUF"}, "filename": {"m.gguf"}}
	if rec := hfhPost(s, form, true); rec.Code != http.StatusOK {
		t.Fatalf("start: status %d\n%s", rec.Code, rec.Body.String())
	}
	id := "org--Model-GGUF--m"
	testutil.Eventually(t, func() bool { return host.gets.Load() == 1 }, "the download never reached the host")

	if rec := hfhProgress(s, id, true); rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "<progress") {
		t.Errorf("while downloading: status %d body %q, want 200 with a progress bar", rec.Code, rec.Body.String())
	}

	host.unblock()
	testutil.Eventually(t, func() bool {
		rec := hfhProgress(s, id, true)
		return rec.Code == htmxStopPolling && strings.Contains(rec.Body.String(), "Download complete!")
	}, "the finished download did not stop the polling")

	// The stream for other clients ends with a done event.
	if body := hfhProgress(s, id, false).Body.String(); !strings.HasSuffix(body, "event: done\ndata: complete\n\n") {
		t.Errorf("stream did not end with the done event: %q", body)
	}

	if rec := hfhProgress(s, "no-such-download", true); rec.Code != htmxStopPolling {
		t.Errorf("unknown download: status %d, want %d", rec.Code, htmxStopPolling)
	}
}
