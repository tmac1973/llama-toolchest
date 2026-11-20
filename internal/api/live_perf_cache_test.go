package api

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/tmac1973/llama-toolchest/internal/benchmark"
)

// waitTimings waits for the timing capture, which runs in its own
// goroutine, to record a sample for model.
func waitTimings(t *testing.T, s *Server, model string, n int) []benchmark.TimingSample {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for {
		got := s.bench.Timings(model)
		if len(got) >= n || time.Now().After(deadline) {
			return got
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// A fork instance's chats reach Live Performance under the instance
// name, streamed or not, with the prompt cache use.
func TestForkChatsReachLivePerformance(t *testing.T) {
	s, _ := runningFork(t)
	proxyPost(s, "/v1/chat/completions", `{"model":"qwen-next","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	got := waitTimings(t, s, "qwen-next", 1)
	if len(got) != 1 {
		t.Fatalf("streamed chat: %d samples, want 1", len(got))
	}
	if got[0].PromptTokens != 10 || got[0].GenTokens != 5 || got[0].CacheTokens == nil || *got[0].CacheTokens != 30 {
		t.Errorf("sample = %+v (cache %v)", got[0], got[0].CacheTokens)
	}

	rec := proxyPost(s, "/v1/chat/completions", `{"model":"qwen-next","messages":[{"role":"user","content":"hi"}]}`)
	if !strings.Contains(rec.Body.String(), `"content":"hi"`) {
		t.Fatalf("the client did not get the JSON answer: %s", rec.Body)
	}
	if got := waitTimings(t, s, "qwen-next", 2); len(got) != 2 {
		t.Fatalf("JSON chat: %d samples, want 2", len(got))
	}

	// Other paths are not chats and are not recorded.
	proxyPost(s, "/v1/completions", `{"model":"qwen-next","prompt":"x"}`)
	time.Sleep(50 * time.Millisecond)
	if got := s.bench.Timings("qwen-next"); len(got) != 2 {
		t.Errorf("a completions request was recorded: %d samples", len(got))
	}
}

// The router's chats keep cache_n too; an engine without it leaves it nil.
func TestCaptureTimingsReadsCacheN(t *testing.T) {
	s := newTestServer(t)
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	s.captureTimings("m", map[string]any{"prompt_n": 12.0, "cache_n": 400.0, "predicted_n": 3.0,
		"prompt_per_second": 100.0, "predicted_per_second": 10.0})
	s.captureTimings("m", map[string]any{"prompt_n": 12.0, "predicted_n": 3.0,
		"prompt_per_second": 100.0, "predicted_per_second": 10.0})
	got := s.bench.Timings("m")
	if len(got) != 2 || got[0].CacheTokens == nil || *got[0].CacheTokens != 400 || got[1].CacheTokens != nil {
		t.Fatalf("samples = %+v", got)
	}
}

func TestTimingsResetClearsOneRow(t *testing.T) {
	s := newTestServer(t)
	s.bench = benchmark.NewStore(t.TempDir(), nil)
	for _, m := range []string{"qwen", "gemma"} {
		s.captureTimings(m, map[string]any{"prompt_n": 12.0, "cache_n": 36.0, "predicted_n": 3.0,
			"prompt_per_second": 100.0, "predicted_per_second": 10.0})
	}

	// The table shows the cache column and a Reset per row.
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/timings", nil)
	req.Header.Set("HX-Request", "true")
	s.handleTimings(rec, req)
	body := rec.Body.String()
	for _, want := range []string{"<strong>75%</strong> <small>· 1/1</small>", `name="model" value="qwen"`, `name="model" value="gemma"`, `hx-post="/api/timings/reset"`} {
		if !strings.Contains(body, want) {
			t.Errorf("table lacks %q:\n%s", want, body)
		}
	}

	form := url.Values{"model": {"qwen"}}
	req = httptest.NewRequest("POST", "/api/timings/reset", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec = httptest.NewRecorder()
	s.handleTimingsReset(rec, req)
	if rec.Code != http.StatusOK || strings.Contains(rec.Body.String(), `value="qwen"`) || !strings.Contains(rec.Body.String(), `value="gemma"`) {
		t.Errorf("reset: %d %s", rec.Code, rec.Body)
	}
	if len(s.bench.Timings("qwen")) != 0 || len(s.bench.Timings("gemma")) != 1 {
		t.Error("reset cleared the wrong samples")
	}

	rec = httptest.NewRecorder()
	s.handleTimingsReset(rec, httptest.NewRequest("POST", "/api/timings/reset", nil))
	if rec.Code != http.StatusBadRequest {
		t.Errorf("reset without a model = %d, want 400", rec.Code)
	}
}
